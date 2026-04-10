import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel

from config import (
    BERT_MODEL, DROPOUT, FOCAL_GAMMA, CONTRASTIVE_TAU, CONTRASTIVE_W,
    EVASION_LABELS, STANCE_LABELS,
)


# ════════════════════════════════════════════════════════════════════════════
# Utility
# ════════════════════════════════════════════════════════════════════════════

def mean_pooling(h, mask):
    m = mask.unsqueeze(-1).expand(h.size()).float()
    return torch.sum(h * m, 1) / torch.clamp(m.sum(1), min=1e-9)


# ════════════════════════════════════════════════════════════════════════════
# Focal Loss  (Change 2)
# ════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    NOTE: No label smoothing — mixing smoothing with focal loss destabilises
    p_t computation (smoothed targets reduce p_t artificially).
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        p   = F.softmax(logits, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = torch.clamp(p_t, min=1e-8)
        focal_w = (1.0 - p_t) ** self.gamma
        ce_loss = -torch.log(p_t)
        loss    = focal_w * ce_loss
        if self.weight is not None:
            alpha = self.weight[targets]
            loss  = loss * alpha
        return loss.mean()


# ════════════════════════════════════════════════════════════════════════════
# Token-Level Cross-Attention  (Change 4)
# ════════════════════════════════════════════════════════════════════════════

class TokenCrossAttention(nn.Module):
    """
    Q tokens attend over full answer hidden states → [B, H] summary.
    Stores last_attn_weights [B, Q, T] for explanation extraction.
    """
    def __init__(self, H, n_heads=4, dropout=0.1):
        super().__init__()
        assert H % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = H // n_heads
        self.Wq = nn.Linear(H, H); self.Wk = nn.Linear(H, H); self.Wv = nn.Linear(H, H)
        self.out_proj = nn.Linear(H, H)
        self.norm     = nn.LayerNorm(H)
        self.drop     = nn.Dropout(dropout)
        self.last_attn_weights = None   # set during forward; used by inference

    def forward(self, q_seq, a_seq, a_mask, q_mask=None):
        B, Q, H = q_seq.size()
        T       = a_seq.size(1)
        Q_ = self.Wq(q_seq).view(B, Q, self.n_heads, self.d_head).transpose(1, 2)
        K_ = self.Wk(a_seq).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V_ = self.Wv(a_seq).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scale  = self.d_head ** 0.5
        scores = torch.matmul(Q_, K_.transpose(-2, -1)) / scale
        if a_mask is not None:
            pad_mask = (a_mask == 0).unsqueeze(1).unsqueeze(2)
            scores   = scores.masked_fill(pad_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        self.last_attn_weights = attn.mean(dim=1).detach()   # [B, Q, T]
        ctx  = torch.matmul(attn, V_)
        ctx  = ctx.transpose(1, 2).contiguous().view(B, Q, H)
        ctx  = self.out_proj(ctx)
        if q_mask is not None:
            qm  = q_mask.unsqueeze(-1).float()
            out = (ctx * qm).sum(1) / qm.sum(1).clamp(min=1e-9)
        else:
            out = ctx.mean(1)
        return self.norm(out)


# ════════════════════════════════════════════════════════════════════════════
# Token-Level Gating  (V3-C base — Change 1)
# ════════════════════════════════════════════════════════════════════════════

class TokenLevelGating(nn.Module):
    """
    Each answer token is gated by the question representation,
    then mean-pooled to [B, H].
    """
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


# ════════════════════════════════════════════════════════════════════════════
# Contrastive Loss  (Change 7)
# ════════════════════════════════════════════════════════════════════════════

class ContrastiveLoss(nn.Module):
    """
    Supervised NT-Xent: same-label reps pulled together,
    different-label reps pushed apart (evasion batches only).
    NaN-safe: uses manual norm clamp instead of F.normalize.
    """
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, embeddings, labels):
        B = embeddings.size(0)
        if B < 2: return torch.tensor(0.0, device=embeddings.device)
        norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        z     = embeddings / norms
        sim   = torch.matmul(z, z.T) / self.tau
        mask_diag = torch.eye(B, dtype=torch.bool, device=z.device)
        sim   = sim.masked_fill(mask_diag, float("-inf"))
        label_mat = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask  = label_mat & ~mask_diag
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        log_probs = F.log_softmax(sim, dim=-1)
        log_probs = torch.clamp(log_probs, min=-100)
        loss      = -(log_probs * pos_mask.float()).sum() / pos_mask.float().sum()
        if torch.isnan(loss):
            return torch.tensor(0.0, device=embeddings.device)
        return loss


# ════════════════════════════════════════════════════════════════════════════
# Classification head
# ════════════════════════════════════════════════════════════════════════════

class ClassHead(nn.Module):
    """
    Optional log(answer_length) feature for the evasion head.
    Helps the model not conflate response verbosity with evasiveness.
    """
    def __init__(self, H, n, dr, use_len_feat=False):
        super().__init__()
        self.use_len_feat = use_len_feat
        in_dim = H + 1 if use_len_feat else H
        self.net = nn.Sequential(
            nn.Dropout(dr), nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Dropout(dr), nn.Linear(256, n))

    def forward(self, x, len_feat=None):
        if self.use_len_feat and len_feat is not None:
            x = torch.cat([x, len_feat], dim=-1)
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════════
# Main model
# ════════════════════════════════════════════════════════════════════════════

class MultiTaskV3C(nn.Module):
    """
    V3-C + All 7 changes:
      [1] Base: TokenLevelGating (V3-C ablation winner)
      [2] FocalLoss(gamma=3) for evasion
      [4] TokenCrossAttention replaces mean-pooled question repr
      [7] ContrastiveLoss on gated reps (evasion batches only)
      [FIX] log(answer_length) feature in evasion head

    Forward inputs:
      input_ids / attention_mask : full [Q;A] tokens
      q_input_ids / q_attention_mask : question-only tokens (Change 4)
      clause_ids / clause_masks / n_clauses : needed for interface compat
      task_id : 0=evasion, 1=stance
      labels  : optional (if provided, loss is returned)
    """
    def __init__(self, ew, sw):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(BERT_MODEL)
        H = self.encoder.config.hidden_size   # 768

        self.token_cross_attn = TokenCrossAttention(H, n_heads=4, dropout=0.1)
        self.evasion_gate     = TokenLevelGating(H, DROPOUT)
        self.stance_gate      = TokenLevelGating(H, DROPOUT)

        self.evasion_head = ClassHead(H, len(EVASION_LABELS), DROPOUT, use_len_feat=True)
        self.stance_head  = ClassHead(H, len(STANCE_LABELS),  DROPOUT, use_len_feat=False)

        self.evasion_loss_fn     = FocalLoss(weight=ew, gamma=FOCAL_GAMMA)
        self.stance_loss_fn      = nn.CrossEntropyLoss(weight=sw, label_smoothing=0.1)
        self.contrastive_loss_fn = ContrastiveLoss(tau=CONTRASTIVE_TAU)

    # ── gradient management ──────────────────────────────────────────────
    def freeze_inactive_head(self, task):
        off = self.stance_head  if task == 0 else self.evasion_head
        on  = self.evasion_head if task == 0 else self.stance_head
        for p in off.parameters(): p.requires_grad = False
        for m in [on, self.encoder, self.token_cross_attn,
                  self.evasion_gate, self.stance_gate]:
            for p in m.parameters(): p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True

    # ── forward ─────────────────────────────────────────────────────────
    def forward(self, input_ids, attention_mask, task_id, labels=None,
                q_input_ids=None, q_attention_mask=None,
                clause_ids=None, clause_masks=None, n_clauses=None):
        task = int(task_id[0].item())

        # 1. Encode full [Q;A]
        h_out = self.encoder(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state   # [B,T,H]

        # 2. Encode question tokens
        q_out = self.encoder(input_ids=q_input_ids,
                             attention_mask=q_attention_mask).last_hidden_state # [B,Q,H]

        # 3. Token cross-attention → rich question repr
        q_rep = self.token_cross_attn(q_out, h_out, attention_mask,
                                       q_attention_mask)                         # [B,H]

        # 4. Gate answer tokens
        gate_fn = self.evasion_gate if task == 0 else self.stance_gate
        gated   = gate_fn(h_out, attention_mask, q_rep)                         # [B,H]

        # 5. Classify
        len_feat = torch.log(attention_mask.float().sum(dim=1, keepdim=True).clamp(min=1))
        logits   = self.evasion_head(gated, len_feat=len_feat) if task == 0 \
                   else self.stance_head(gated)

        loss = None
        if labels is not None:
            cls_loss = (self.evasion_loss_fn if task == 0 else self.stance_loss_fn)(logits, labels)
            if task == 0:
                contr_loss = self.contrastive_loss_fn(gated, labels)
                loss = cls_loss + CONTRASTIVE_W * contr_loss
            else:
                loss = cls_loss

        return loss, logits
