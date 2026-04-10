import argparse, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from config import (
    FULL_MODEL_PATH, MAX_LEN, STANCE_MAX_LEN, Q_MAX_LEN,
    MAX_CLAUSES, CLAUSE_LEN, EVASION_LABELS, STANCE_LABELS,
)
from data import tokenizer, split_into_clauses
from model import MultiTaskV3C


# ════════════════════════════════════════════════════════════════════════════
# Model loader
# ════════════════════════════════════════════════════════════════════════════

def load_model(model_path, device):
    """Load a saved V3-C checkpoint. Dummy weights used since
    FocalLoss / ContrastiveLoss are not called during inference."""
    dummy_ew = torch.ones(3, device=device)
    dummy_sw = torch.ones(2, device=device)
    model = MultiTaskV3C(dummy_ew, dummy_sw).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.unfreeze_all()
    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded model from: {model_path}")
    print(f"  Parameters: {total:,}")
    return model


# ════════════════════════════════════════════════════════════════════════════
# Single-example inference with explanation
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_with_explanation(model, question, answer, device):
    """
    Run evasion + stance prediction on a single Q/A pair.

    Returns dict:
      evasion_pred / evasion_conf
      stance_pred  / stance_conf
      clause_data  : list of (clause_text, relevance_score, mismatch_score)
      top_clause   : the most explanatory clause (class-conditional)
      expl_note    : human-readable note describing what top_clause shows
      token_attn   : [Q_len, T_len] cross-attention weights (for visualization)
    """
    model.eval()

    # ── Tokenise full [Q;A] ──────────────────────────────────────────────
    enc  = tokenizer(question, answer, truncation=True,
                     padding="max_length", max_length=MAX_LEN,
                     return_tensors="pt").to(device)
    qenc = tokenizer(question, truncation=True,
                     padding="max_length", max_length=Q_MAX_LEN,
                     return_tensors="pt").to(device)

    # ── Clause tensors ───────────────────────────────────────────────────
    clauses = split_into_clauses(answer)
    ids_l, masks_l = [], []
    for c in clauses[:MAX_CLAUSES]:
        ce = tokenizer(question, c, truncation=True,
                       padding="max_length", max_length=CLAUSE_LEN,
                       return_tensors="pt").to(device)
        ids_l.append(ce["input_ids"]); masks_l.append(ce["attention_mask"])
    n_real = len(ids_l)
    while len(ids_l) < MAX_CLAUSES:
        ids_l.append(torch.zeros(1, CLAUSE_LEN, dtype=torch.long, device=device))
        masks_l.append(torch.zeros(1, CLAUSE_LEN, dtype=torch.long, device=device))
    clause_ids   = torch.cat(ids_l).unsqueeze(0)
    clause_masks = torch.cat(masks_l).unsqueeze(0)
    n_clauses    = torch.tensor([n_real], dtype=torch.long, device=device)

    # ── Evasion forward ──────────────────────────────────────────────────
    _, e_logits = model(
        enc["input_ids"], enc["attention_mask"],
        torch.tensor([0], device=device),
        q_input_ids=qenc["input_ids"], q_attention_mask=qenc["attention_mask"],
        clause_ids=clause_ids, clause_masks=clause_masks, n_clauses=n_clauses)
    evasion_pred = EVASION_LABELS[torch.argmax(e_logits, 1).item()]
    evasion_conf = float(torch.softmax(e_logits, 1).max())

    # Retrieve cross-attention weights (set as side-effect during evasion forward)
    token_attn = model.token_cross_attn.last_attn_weights[0].cpu().numpy()   # [Q, T]

    # ── Stance forward ───────────────────────────────────────────────────
    enc_s  = tokenizer(question, answer, truncation=True,
                       padding="max_length", max_length=STANCE_MAX_LEN,
                       return_tensors="pt").to(device)
    qenc_s = tokenizer(question, truncation=True,
                       padding="max_length", max_length=Q_MAX_LEN,
                       return_tensors="pt").to(device)
    _, s_logits = model(
        enc_s["input_ids"], enc_s["attention_mask"],
        torch.tensor([1], device=device),
        q_input_ids=qenc_s["input_ids"], q_attention_mask=qenc_s["attention_mask"],
        clause_ids=clause_ids, clause_masks=clause_masks, n_clauses=n_clauses)
    stance_pred = STANCE_LABELS[torch.argmax(s_logits, 1).item()]
    stance_conf = float(torch.softmax(s_logits, 1).max())

    # ── Clause-level explanation (cosine mismatch signal) ────────────────
    relevance_scores, mismatch_scores = [], []
    for clause in clauses:
        ce = tokenizer(question, clause, truncation=True,
                       padding="max_length", max_length=CLAUSE_LEN,
                       return_tensors="pt").to(device)
        q_out = model.encoder(input_ids=qenc["input_ids"],
                              attention_mask=qenc["attention_mask"]).last_hidden_state
        a_out = model.encoder(input_ids=ce["input_ids"],
                              attention_mask=ce["attention_mask"]).last_hidden_state
        q_mean = q_out[0, :qenc["attention_mask"].sum().item()].mean(0)
        a_mean = a_out[0, :ce["attention_mask"].sum().item()].mean(0)
        relevance = float(F.cosine_similarity(q_mean.unsqueeze(0), a_mean.unsqueeze(0)))
        mismatch  = 1.0 - max(0.0, relevance)
        relevance_scores.append(relevance)
        mismatch_scores.append(mismatch)

    rel_arr = np.array(relevance_scores, dtype=float)
    mis_arr = np.array(mismatch_scores,  dtype=float)
    rel_arr = rel_arr / (rel_arr.sum() + 1e-9)
    mis_arr = mis_arr / (mis_arr.sum() + 1e-9)
    clause_data = list(zip(clauses, rel_arr.tolist(), mis_arr.tolist()))

    # ── Class-conditional top-clause selection ───────────────────────────
    if evasion_pred == "Non-Evasive":
        top_idx    = int(np.argmax(rel_arr))
        top_clause = clauses[top_idx]
        expl_note  = "Most relevant clause (high Q-A alignment):"
    elif evasion_pred == "Partially Evasive":
        rel_idx      = int(np.argmax(rel_arr))
        mis_idx      = int(np.argmax(mis_arr))
        top_clause   = clauses[rel_idx]
        drift_clause = clauses[mis_idx] if mis_idx != rel_idx else "(same clause)"
        expl_note    = f"Partial answer clause | Drift clause: {drift_clause[:100]}"
    else:   # Evasive
        top_idx    = int(np.argmax(mis_arr))
        top_clause = clauses[top_idx]
        expl_note  = "Most evasive clause (high Q-A mismatch):"

    return {
        "evasion_pred": evasion_pred,
        "evasion_conf": evasion_conf,
        "stance_pred":  stance_pred,
        "stance_conf":  stance_conf,
        "clause_data":  clause_data,
        "top_clause":   top_clause,
        "expl_note":    expl_note,
        "token_attn":   token_attn,
    }


# ════════════════════════════════════════════════════════════════════════════
# Batch inference
# ════════════════════════════════════════════════════════════════════════════

def predict_batch(model, questions, answers, device):
    results = []
    for q, a in tqdm(zip(questions, answers), total=len(questions), desc="Predicting"):
        try:
            results.append(predict_with_explanation(model, q, a, device))
        except Exception as e:
            results.append({"evasion_pred": "Error", "stance_pred": "Error",
                            "evasion_conf": 0.0, "stance_conf": 0.0,
                            "top_clause": "", "expl_note": str(e), "clause_data": []})
    return results


def results_to_dataframe(questions, answers, results, true_evasion=None):
    rows = []
    for i, (q, a, r) in enumerate(zip(questions, answers, results)):
        row = {
            "question":     q,
            "answer":       a,
            "pred_evasion": r.get("evasion_pred", "Error"),
            "evasion_conf": round(r.get("evasion_conf", 0.0), 4),
            "pred_stance":  r.get("stance_pred",  "Error"),
            "stance_conf":  round(r.get("stance_conf",  0.0), 4),
            "top_clause":   r.get("top_clause",   ""),
            "expl_note":    r.get("expl_note",    ""),
            "clause_details": str([(c[:80], round(rv, 4), round(mv, 4))
                                   for c, rv, mv in r.get("clause_data", [])]),
        }
        if true_evasion is not None:
            row["true_evasion"] = true_evasion[i]
        rows.append(row)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def interactive_demo(model, device):
    print("\n" + "=" * 60)
    print("  Interactive inference. Type 'quit' to exit.")
    print("=" * 60)
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("quit", "exit", "q"): break
        answer   = input("Answer:   ").strip()
        if not question or not answer:
            print("  (both fields required)"); continue
        r = predict_with_explanation(model, question, answer, device)
        print(f"\n  Q: {question}")
        print(f"  A: {answer}")
        print(f"\n  Evasion : {r['evasion_pred']} ({r['evasion_conf']*100:.0f}%)")
        print(f"  Stance  : {r['stance_pred']}  ({r['stance_conf']*100:.0f}%)")
        print(f"  {r['expl_note']}")
        print(f"  => {r['top_clause']}")
        print(f"\n  All clauses (relevance / mismatch):")
        for c, rel, mis in r["clause_data"]:
            print(f"    [{rel:.3f} rel / {mis:.3f} mis] {c}")


def main():
    parser = argparse.ArgumentParser(description="V3-C Inference")
    parser.add_argument("--model",  default=FULL_MODEL_PATH,
                        help="Path to saved .pt checkpoint")
    parser.add_argument("--input",  default=None,
                        help="CSV with columns 'question','answer' for batch prediction")
    parser.add_argument("--output", default="predictions.csv",
                        help="Output CSV path for batch predictions")
    parser.add_argument("--demo",   action="store_true",
                        help="Run 5 validation examples from the dataset as demo")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.model}\n"
            f"Run train.py first to generate the checkpoint.")

    model = load_model(args.model, device)

    # ── Batch CSV mode ───────────────────────────────────────────────────
    if args.input:
        df = pd.read_csv(args.input)
        assert "question" in df.columns and "answer" in df.columns, \
            "Input CSV must have 'question' and 'answer' columns."
        print(f"\nRunning batch inference on {len(df)} rows...")
        results = predict_batch(model, df["question"].tolist(),
                                df["answer"].tolist(), device)
        true_ev = df["evasion_label"].tolist() if "evasion_label" in df.columns else None
        out_df  = results_to_dataframe(df["question"].tolist(), df["answer"].tolist(),
                                       results, true_evasion=true_ev)
        out_df.to_csv(args.output, index=False)
        print(f"Saved {len(out_df)} predictions to {args.output}")

    # ── Demo mode (loads dataset) ────────────────────────────────────────
    elif args.demo:
        print("\nLoading validation set for demo...")
        from data import load_all_data
        _, raw, _ = load_all_data()
        Qeva, Aeva, yeva = raw[3], raw[4], raw[5]
        questions = Qeva[:5]; answers = Aeva[:5]
        true_lbls = [EVASION_LABELS[y] for y in yeva[:5]]
        print("\nRunning inference on 5 validation examples:")
        print("─" * 60)
        for i, (q, a, true_lbl) in enumerate(zip(questions, answers, true_lbls)):
            r = predict_with_explanation(model, q, a, device)
            correct = "✓" if r["evasion_pred"] == true_lbl else "✗"
            print(f"\nExample {i+1} {correct}")
            print(f"  Q: {q}")
            print(f"  A: {a}")
            print(f"  True:  {true_lbl}")
            print(f"  Pred:  {r['evasion_pred']} ({r['evasion_conf']*100:.0f}%) | "
                  f"Stance: {r['stance_pred']} ({r['stance_conf']*100:.0f}%)")
            print(f"  {r['expl_note']}")
            print(f"  => {r['top_clause']}")
            print(f"\n  All clauses (relevance / mismatch):")
            for c, rel, mis in r["clause_data"]:
                print(f"    [{rel:.3f} rel / {mis:.3f} mis] {c}")
        print("\n" + "─" * 60)

    # ── Interactive mode ─────────────────────────────────────────────────
    else:
        interactive_demo(model, device)


if __name__ == "__main__":
    main()