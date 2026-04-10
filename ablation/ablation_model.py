import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel

from ablation_config import BERT_MODEL, DROPOUT, EVASION_LABELS, STANCE_LABELS


# ══════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════

def mean_pooling(h, mask):
    m = mask.unsqueeze(-1).expand(h.size()).float()
    return torch.sum(h * m, 1) / torch.clamp(m.sum(1), min=1e-9)


# ══════════════════════════════════════════════════════════════════════════
# Component modules
# ══════════════════════════════════════════════════════════════════════════

class TokenLevelGating(nn.Module):
    """Gate each answer token by the mean-pooled question representation."""
    def __init__(self, H, dr):
        super().__init__()
        self.Wa   = nn.Linear(H, H, bias=False)
        self.Wq   = nn.Linear(H, H, bias=True)
        self.drop = nn.Dropout(dr)

    def forward(self, h_seq, attention_mask, q_rep):
        gate    = torch.sigmoid((self.Wa(h_seq) + self.Wq(q_rep).unsqueeze(1)) / 0.7)
        h_gated = self.drop(gate * h_seq)
        m       = attention_mask.unsqueeze(-1).expand(h_gated.size()).float()
        return torch.sum(h_gated * m, 1) / torch.clamp(m.sum(1), min=1e-9)


class CrossAttention(nn.Module):
    """
    Question mean-pool attends over clause representations.
    Returns a single [B, H] context vector.
    NOTE: uses mean-pooled q_rep (not token-level) — different from final model.
    """
    def __init__(self, H):
        super().__init__()
        self.Wq = nn.Linear(H, H)
        self.Wk = nn.Linear(H, H)
        self.Wv = nn.Linear(H, H)

    def forward(self, q, clause_reps, n_clauses):
        B, C, H = clause_reps.size()
        Q = self.Wq(q).unsqueeze(1)                                    # [B, 1, H]
        K = self.Wk(clause_reps)                                       # [B, C, H]
        V = self.Wv(clause_reps)                                       # [B, C, H]
        scale  = torch.sqrt(torch.tensor(H, dtype=torch.float32, device=q.device))
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale              # [B, 1, C]
        # Mask padding clauses
        valid  = torch.arange(C, device=q.device).unsqueeze(0) < n_clauses.unsqueeze(1)
        scores = scores.squeeze(1).masked_fill(~valid, float("-inf"))  # [B, C]
        weights = F.softmax(scores, dim=-1)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)
        weights = F.dropout(weights, p=0.1, training=self.training)
        return torch.bmm(weights.unsqueeze(1), V).squeeze(1)           # [B, H]


class GraphReasoningLayer(nn.Module):
    """
    2-round message passing over clause nodes.
    Each node aggregates exclude-self neighbour mean, then updates via MLP + residual.
    """
    def __init__(self, H, dr, rounds=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(H * 2, H), nn.ReLU(), nn.Dropout(dr))
            for _ in range(rounds)
        ])
        self.norm = nn.LayerNorm(H)

    def forward(self, x, n_clauses):
        B, C, H = x.size()
        valid = (torch.arange(C, device=x.device).unsqueeze(0)
                 < n_clauses.unsqueeze(1)).float()                     # [B, C]
        for layer in self.layers:
            sum_x  = (x * valid.unsqueeze(-1)).sum(1, keepdim=True)    # [B, 1, H]
            denom  = (n_clauses.unsqueeze(-1).unsqueeze(-1) - 1).clamp(min=1)
            nbr_mean = (sum_x - x * valid.unsqueeze(-1)) / denom       # [B, C, H]
            x = self.norm(x + layer(torch.cat([x, nbr_mean], dim=-1)) * valid.unsqueeze(-1))
        return x


class ClauseAttentionPooling(nn.Module):
    """Weighted sum of clause reps using learned scalar attention scores."""
    def __init__(self, H, dr):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, 1))
        self.norm  = nn.LayerNorm(H)
        self.drop  = nn.Dropout(dr)

    def forward(self, x, n_clauses):
        B, C, _ = x.size()
        valid = (torch.arange(C, device=x.device).unsqueeze(0) < n_clauses.unsqueeze(1))
        s = self.score(x).squeeze(-1).masked_fill(~valid, float("-inf"))  # [B, C]
        w = F.softmax(s, dim=-1).unsqueeze(-1)                             # [B, C, 1]
        return self.norm(self.drop((w * x).sum(1)))                        # [B, H]


class ClassHead(nn.Module):
    def __init__(self, H, n, dr):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dr), nn.Linear(H, 256), nn.ReLU(),
            nn.Dropout(dr), nn.Linear(256, n))

    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════════════
# Main configurable model
# ══════════════════════════════════════════════════════════════════════════

class MultiTaskDistilBERT_V3(nn.Module):
    """
    Configurable V3 for ablation study.

    Flags (all True = full V3):
      use_gating      : TokenLevelGating on the full answer sequence
      use_cross_attn  : CrossAttention of question over clause reps
      use_graph       : GraphReasoningLayer message passing between clauses
      use_clause_pool : ClauseAttentionPooling weighted sum of clause reps

    Fusion: base + 0.7*xattn_out (if active) + 0.7*cl_out (if active)
            -> LayerNorm -> Dropout(0.2) -> ClassHead

    Fallback degradation:
      cross+graph+pool=False -> gating-only (~ V2)
      gating=False too       -> plain mean-pool (~ V1)
    """
    def __init__(self, ew, sw,
                 use_gating=True, use_cross_attn=True,
                 use_graph=True,  use_clause_pool=True):
        super().__init__()
        self.use_gating      = use_gating
        self.use_cross_attn  = use_cross_attn
        self.use_graph       = use_graph
        self.use_clause_pool = use_clause_pool

        self.encoder     = DistilBertModel.from_pretrained(BERT_MODEL)
        H = self.encoder.config.hidden_size   # 768

        self.clause_proj = nn.Linear(H, H)

        if use_cross_attn:
            self.cross_attn = CrossAttention(H)
        if use_graph:
            self.graph = GraphReasoningLayer(H, DROPOUT, rounds=2)
        if use_clause_pool:
            self.cl_pool = ClauseAttentionPooling(H, DROPOUT)
        if use_gating:
            self.evasion_gate = TokenLevelGating(H, DROPOUT)
            self.stance_gate  = TokenLevelGating(H, DROPOUT)

        self.fusion       = nn.LayerNorm(H)
        self.evasion_head = ClassHead(H, len(EVASION_LABELS), DROPOUT)
        self.stance_head  = ClassHead(H, len(STANCE_LABELS),  DROPOUT)

        # Plain CrossEntropyLoss (no Focal, no Contrastive — ablation study)
        self.evasion_loss_fn = nn.CrossEntropyLoss(weight=ew)
        self.stance_loss_fn  = nn.CrossEntropyLoss(weight=sw)

    # ── Clause encoding ─────────────────────────────────────────────────
    def _encode_clauses(self, clause_ids, clause_masks, B, C):
        ids   = clause_ids.view(B * C, -1)
        masks = clause_masks.view(B * C, -1)
        valid = masks.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.zeros(B, C, self.encoder.config.hidden_size, device=ids.device)
        valid_h = self.encoder(input_ids=ids[valid],
                               attention_mask=masks[valid]).last_hidden_state
        pooled  = mean_pooling(valid_h, masks[valid])
        pool    = torch.zeros(ids.size(0), pooled.size(-1), device=ids.device)
        pool[valid] = pooled
        pool = F.relu(self.clause_proj(pool))
        return pool.view(B, C, -1)

    # ── Gradient management ──────────────────────────────────────────────
    def freeze_inactive_head(self, task):
        """Freeze the off-task head/gate; unfreeze on-task head/gate + shared."""
        task_modules = [
            [self.evasion_head] + ([self.evasion_gate] if self.use_gating else []),
            [self.stance_head]  + ([self.stance_gate]  if self.use_gating else []),
        ]
        shared = [self.encoder, self.clause_proj, self.fusion]
        if self.use_cross_attn:  shared.append(self.cross_attn)
        if self.use_graph:       shared.append(self.graph)
        if self.use_clause_pool: shared.append(self.cl_pool)

        for p in task_modules[1 - task][0].parameters(): p.requires_grad = False
        if len(task_modules[1 - task]) > 1:
            for p in task_modules[1 - task][1].parameters(): p.requires_grad = False
        for p in task_modules[task][0].parameters(): p.requires_grad = True
        if len(task_modules[task]) > 1:
            for p in task_modules[task][1].parameters(): p.requires_grad = True
        for m in shared:
            for p in m.parameters(): p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True

    # ── Forward ─────────────────────────────────────────────────────────
    def forward(self, input_ids, attention_mask, task_id, labels=None,
                q_input_ids=None, q_attention_mask=None,
                clause_ids=None, clause_masks=None, n_clauses=None):
        B    = input_ids.size(0)
        task = int(task_id[0].item())

        # 1. Global encoding
        h     = self.encoder(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state      # [B, T, H]
        q_h   = self.encoder(input_ids=q_input_ids,
                             attention_mask=q_attention_mask).last_hidden_state    # [B, Q, H]
        q_rep = F.normalize(mean_pooling(q_h, q_attention_mask), dim=-1)          # [B, H]

        # 2. Base: gated or plain mean pool
        if self.use_gating:
            gate_fn = self.evasion_gate if task == 0 else self.stance_gate
            base    = gate_fn(h, attention_mask, q_rep)
        else:
            base = mean_pooling(h, attention_mask)

        # 3. Clause pipeline
        C           = clause_ids.size(1)
        clause_reps = self._encode_clauses(clause_ids, clause_masks, B, C)        # [B, C, H]

        xattn_out = self.cross_attn(q_rep, clause_reps, n_clauses) \
                    if self.use_cross_attn else None

        if self.use_graph:
            clause_reps = self.graph(clause_reps, n_clauses)

        cl_out = self.cl_pool(clause_reps, n_clauses) if self.use_clause_pool else None

        # 4. Additive fusion
        fused = base
        if xattn_out is not None: fused = fused + 0.7 * xattn_out
        if cl_out    is not None: fused = fused + 0.7 * cl_out
        fused = self.fusion(F.dropout(fused, p=0.2, training=self.training))

        # 5. Classify
        head_fn = self.evasion_head if task == 0 else self.stance_head
        logits  = head_fn(fused)

        loss = None
        if labels is not None:
            loss_fn = self.evasion_loss_fn if task == 0 else self.stance_loss_fn
            loss    = loss_fn(logits, labels)

        return loss, logits
