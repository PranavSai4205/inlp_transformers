import os, random, itertools, argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from ablation_config import (
    SEED, EPOCHS, FREEZE_EPOCHS, GRAD_ACCUM, WARMUP_RATIO,
    LR_EVASION, LR_STANCE, OUTPUT_DIR,
    EVASION_LABELS, STANCE_LABELS, ABLATION_CONFIGS,
)
from ablation_data import load_all_data
from ablation_model import MultiTaskDistilBERT_V3

# ── Reproducibility ───────────────────────────────────────────────────────
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")


# ══════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, label_names):
    model.eval(); model.unfreeze_all()
    all_preds, all_labels, total_loss = [], [], 0.0
    for batch in loader:
        loss, logits = model(
            batch["input_ids"].to(device),       batch["attention_mask"].to(device),
            batch["task_id"].to(device),          batch["labels"].to(device),
            batch["q_input_ids"].to(device),      batch["q_attention_mask"].to(device),
            batch["clause_ids"].to(device),       batch["clause_masks"].to(device),
            batch["n_clauses"].to(device))
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, 1).cpu().tolist())
        all_labels.extend(batch["labels"].tolist())
    return {
        "loss":   total_loss / len(loader),
        "acc":    accuracy_score(all_labels, all_preds),
        "f1":     f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "prec":   precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "rec":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "report": classification_report(all_labels, all_preds,
                                        target_names=label_names, zero_division=0),
        "cm":     pd.DataFrame(
                      confusion_matrix(all_labels, all_preds,
                                       labels=list(range(len(label_names)))),
                      index=[f"True:{l}" for l in label_names],
                      columns=[f"Pred:{l}" for l in label_names]),
        "preds":  all_preds,
        "labels": all_labels,
    }


def print_eval(m, title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(f"  Loss:{m['loss']:.4f}  Acc:{m['acc']*100:.2f}%  MacroF1:{m['f1']:.4f}  "
          f"Prec:{m['prec']:.4f}  Rec:{m['rec']:.4f}")
    print("\n  Per-class Report:")
    for line in m["report"].strip().split("\n"): print(f"  {line}")
    print("\n  Confusion Matrix:")
    print(m["cm"].to_string())
    print("=" * 60)


def get_pe_f1(report_str):
    for line in report_str.split("\n"):
        if "Partially Evasive" in line:
            parts = line.split()
            try:    return float(parts[3])
            except: return 0.0
    return 0.0


# ══════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════

def train_model(model, loaders, variant_name, save_tag, n_epochs=EPOCHS):
    """
    Train one V3 variant (full or ablated).
    Sampling: 2:1 evasion priority (2 evasion batches per stance batch).
    Scheduler: linear warmup (get_linear_schedule_with_warmup).
    Returns: (e_test, s_test, history)
    """
    best_path = os.path.join(OUTPUT_DIR, f"best_{save_tag}.pt")

    # ── Optimizer ───────────────────────────────────────────────────────
    shared_params = (list(model.encoder.parameters()) +
                     list(model.clause_proj.parameters()) +
                     list(model.fusion.parameters()))
    if hasattr(model, "cross_attn"):  shared_params += list(model.cross_attn.parameters())
    if hasattr(model, "graph"):       shared_params += list(model.graph.parameters())
    if hasattr(model, "cl_pool"):     shared_params += list(model.cl_pool.parameters())

    ev_task = list(model.evasion_head.parameters())
    if hasattr(model, "evasion_gate"): ev_task += list(model.evasion_gate.parameters())
    st_task = list(model.stance_head.parameters())
    if hasattr(model, "stance_gate"):  st_task += list(model.stance_gate.parameters())

    optimizer = AdamW([
        {"params": ev_task,       "lr": LR_EVASION},
        {"params": st_task,       "lr": LR_STANCE},
        {"params": shared_params, "lr": LR_EVASION},
    ], weight_decay=0.01)

    # 2:1 sampling → 3 batches per stance batch → n_steps multiplier = 3
    n_steps   = len(loaders["stance_train"]) * 3 * n_epochs
    n_warmup  = int(WARMUP_RATIO * n_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, n_warmup, n_steps)

    # Freeze encoder initially
    for p in model.encoder.parameters(): p.requires_grad = False

    best_avg_f1 = 0.0
    history     = []

    print(f"\n{'='*60}")
    print(f"  TRAINING: {variant_name}")
    print(f"{'='*60}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Sampling: 2:1 evasion priority  |  Scheduler: linear warmup")
    print(f"  LR_evasion:{LR_EVASION}  LR_stance:{LR_STANCE}  "
          f"Freeze:{FREEZE_EPOCHS} epochs  Epochs:{n_epochs}")
    print(f"  Checkpoint: {best_path}")

    for epoch in range(1, n_epochs + 1):
        if epoch == FREEZE_EPOCHS + 1:
            for p in model.encoder.parameters(): p.requires_grad = True
            print(f"\n  Epoch {epoch}: BERT UNFROZEN.")

        model.train()
        run_e_loss = run_s_loss = 0.0
        n_e = n_s = 0

        # Build 2:1 combined batch list
        e_cycle  = itertools.cycle(list(loaders["evasion_train"]))
        combined = []
        for s_batch in loaders["stance_train"]:
            combined.append(next(e_cycle))
            combined.append(next(e_cycle))
            combined.append(s_batch)

        total = len(combined)
        print(f"\n  Epoch {epoch}/{n_epochs} — {total} steps")
        optimizer.zero_grad()

        for step, batch in enumerate(combined, 1):
            task = int(batch["task_id"][0].item())
            model.freeze_inactive_head(task)
            loss, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["task_id"].to(device),
                batch["labels"].to(device),
                batch["q_input_ids"].to(device),
                batch["q_attention_mask"].to(device),
                batch["clause_ids"].to(device),
                batch["clause_masks"].to(device),
                batch["n_clauses"].to(device))

            (loss / GRAD_ACCUM).backward()
            if task == 0: run_e_loss += loss.item(); n_e += 1
            else:         run_s_loss += loss.item(); n_s += 1

            if step % GRAD_ACCUM == 0 or step == total:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            # Progress bar
            done = int(30 * step / total)
            print(f"\r  [{'='*done}{'-'*(30-done)}] {step}/{total}", end="", flush=True)

        print()
        e_val   = evaluate(model, loaders["evasion_val"], EVASION_LABELS)
        s_val   = evaluate(model, loaders["stance_val"],  STANCE_LABELS)
        avg_f1  = (e_val["f1"] + s_val["f1"]) / 2
        is_best = avg_f1 > best_avg_f1

        e_loss_avg = run_e_loss / n_e if n_e else 0
        s_loss_avg = run_s_loss / n_s if n_s else 0
        print(f"  E.loss:{e_loss_avg:.4f} S.loss:{s_loss_avg:.4f} | "
              f"Val E.F1:{e_val['f1']:.4f} S.F1:{s_val['f1']:.4f} "
              f"Avg:{avg_f1:.4f} E.ValLoss:{e_val['loss']:.4f}  "
              f"{'<-- BEST' if is_best else ''}")
        print("  Val per-class evasion:")
        for line in e_val["report"].strip().split("\n"):
            if any(lbl in line for lbl in EVASION_LABELS):
                print(f"    {line.strip()}")

        if is_best:
            best_avg_f1 = avg_f1
            torch.save(model.state_dict(), best_path)
        history.append({
            "epoch": epoch, "e_f1": e_val["f1"], "s_f1": s_val["f1"],
            "avg_f1": avg_f1, "e_val_loss": e_val["loss"],
        })
        print()

    # Reload best and evaluate on test set
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.unfreeze_all()
    e_test = evaluate(model, loaders["evasion_test"], EVASION_LABELS)
    s_test = evaluate(model, loaders["stance_test"],  STANCE_LABELS)
    avg    = (e_test["f1"] + s_test["f1"]) / 2
    print(f"\n  BEST TEST — E.F1:{e_test['f1']:.4f}  S.F1:{s_test['f1']:.4f}  "
          f"Avg:{avg:.4f}  Acc:{e_test['acc']*100:.2f}%")
    return e_test, s_test, history


# ══════════════════════════════════════════════════════════════════════════
# Results table
# ══════════════════════════════════════════════════════════════════════════

def print_results_table(rows):
    print("\n" + "=" * 70)
    print("  ABLATION STUDY — FINAL RESULTS TABLE")
    print("=" * 70)
    print(f"  {'Variant':<40} {'Eva.F1':>8} {'Sta.F1':>8} {'Avg.F1':>8} {'Eva.Acc':>9}")
    print(f"  {'-' * 70}")
    best_avg = max(r[3] for r in rows)
    for row in rows:
        mark = " <-- BEST" if abs(row[3] - best_avg) < 1e-6 else ""
        print(f"  {row[0]:<40} {row[1]:>8.4f} {row[2]:>8.4f} "
              f"{row[3]:>8.4f} {row[4]*100:>8.2f}%{mark}")
    print(f"  {'=' * 70}")

    print()
    print("  PARTIALLY EVASIVE F1 BREAKDOWN (minority class — most critical):")
    print(f"  {'Variant':<40} {'PE F1':>8}")
    print(f"  {'-' * 52}")
    for row in rows:
        print(f"  {row[0]:<40} {row[5]:>8.4f}")
    print("=" * 70)

    best_row = max(rows, key=lambda r: r[3])
    print(f"\n  KEY FINDING:")
    print(f"    Best variant: {best_row[0]}")
    print(f"    => Avg Macro F1: {best_row[3]:.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

VARIANT_KEYS = {
    "full": "full",
    "v3a":  "V3-A: No Graph Reasoning",
    "v3b":  "V3-B: No Cross-Attention",
    "v3c":  "V3-C: No Clause Hierarchy (Gating only)",
    "v3d":  "V3-D: No Gating",
}


def main():
    parser = argparse.ArgumentParser(description="Ablation study for V3 architecture")
    parser.add_argument(
        "--variants", nargs="+",
        default=["full", "v3a", "v3b", "v3c", "v3d"],
        choices=list(VARIANT_KEYS.keys()),
        help="Which variants to run. Default: all. "
             "Options: full v3a v3b v3c v3d. "
             "Example: --variants v3c v3d  (re-runs only V3-C and V3-D)")
    args = parser.parse_args()

    loaders, raw, (ew, sw) = load_all_data()
    Qete, Aete, yete, Qste, Aste, yste = raw

    ew_dev = ew.to(device)
    sw_dev = sw.to(device)

    results_map = {}  # variant_name -> (e_test, s_test)

    # ── Full V3 ──────────────────────────────────────────────────────────
    if "full" in args.variants:
        print("\n" + "=" * 60)
        print("  SECTION 1: FULL V3 + Data Augmentation")
        print("=" * 60)
        model_full = MultiTaskDistilBERT_V3(
            ew_dev, sw_dev,
            use_gating=True, use_cross_attn=True,
            use_graph=True,  use_clause_pool=True
        ).to(device)
        e_full, s_full, _ = train_model(
            model_full, loaders,
            "V3-Full (Gating+CrossAttn+Graph+ClausePool) + Aug",
            "v3_full_aug")
        print_eval(e_full, "EVASION TEST — V3 Full + Aug")
        print_eval(s_full, "STANCE  TEST — V3 Full + Aug")
        results_map["V3 Full + Aug"] = (e_full, s_full)

    # ── Ablation variants ────────────────────────────────────────────────
    requested_names = {VARIANT_KEYS[k] for k in args.variants if k != "full"}

    print("\n" + "=" * 60)
    print("  SECTION 2: ABLATION STUDY")
    print("=" * 60)

    ablation_results = {}
    for cfg in ABLATION_CONFIGS:
        if cfg["name"] not in requested_names:
            print(f"  Skipping {cfg['name']} (not in --variants)")
            continue
        print(f"\n{'─'*60}")
        print(f"  ABLATION: {cfg['name']}")
        print(f"  {cfg['desc']}")
        print(f"{'─'*60}")
        model_abl = MultiTaskDistilBERT_V3(ew_dev, sw_dev, **cfg["flags"]).to(device)
        e_abl, s_abl, hist_abl = train_model(
            model_abl, loaders, cfg["name"], cfg["tag"])
        ablation_results[cfg["name"]] = {
            "e_f1":    e_abl["f1"],
            "s_f1":    s_abl["f1"],
            "avg":     (e_abl["f1"] + s_abl["f1"]) / 2,
            "acc":     e_abl["acc"],
            "report":  e_abl["report"],
            "history": hist_abl,
            "preds":   e_abl["preds"],
            "labels":  e_abl["labels"],
        }
        results_map[cfg["name"]] = (e_abl, s_abl)
        print_eval(e_abl, f"EVASION TEST — {cfg['name']}")

    print("\nAll requested experiments complete.")

    # ── Results table ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SECTION 3: RESULTS TABLE")
    print("=" * 60)

    # rows: (name, e_f1, s_f1, avg, acc, pe_f1)
    rows = []
    if "V3 Full + Aug" in results_map:
        e, s = results_map["V3 Full + Aug"]
        rows.append(("V3 Full + Aug", e["f1"], s["f1"],
                     (e["f1"] + s["f1"]) / 2, e["acc"], get_pe_f1(e["report"])))
    for cfg in ABLATION_CONFIGS:
        if cfg["name"] in ablation_results:
            r = ablation_results[cfg["name"]]
            rows.append((cfg["name"], r["e_f1"], r["s_f1"],
                         r["avg"], r["acc"], get_pe_f1(r["report"])))

    if rows:
        print_results_table(rows)

    # ── Save outputs ─────────────────────────────────────────────────────
    if rows:
        pd.DataFrame([{
            "variant": r[0], "eva_f1": r[1], "sta_f1": r[2],
            "avg_f1":  r[3], "eva_acc": r[4], "pe_f1": r[5]
        } for r in rows]).to_csv(
            os.path.join(OUTPUT_DIR, "v3_ablation_results.csv"), index=False)
        print(f"\n  Saved: v3_ablation_results.csv")

    # Save full V3 test predictions (if it was run)
    if "V3 Full + Aug" in results_map:
        e, s = results_map["V3 Full + Aug"]
        pd.DataFrame({
            "question": Qete, "answer": Aete,
            "true": [EVASION_LABELS[l] for l in e["labels"]],
            "pred": [EVASION_LABELS[p] for p in e["preds"]],
        }).to_csv(os.path.join(OUTPUT_DIR, "results_v3_full_aug_evasion.csv"), index=False)
        pd.DataFrame({
            "target": Qste, "tweet": Aste,
            "true": [STANCE_LABELS[l] for l in s["labels"]],
            "pred": [STANCE_LABELS[p] for p in s["preds"]],
        }).to_csv(os.path.join(OUTPUT_DIR, "results_v3_full_aug_stance.csv"), index=False)
        print(f"  Saved: results_v3_full_aug_evasion.csv")
        print(f"  Saved: results_v3_full_aug_stance.csv")

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
