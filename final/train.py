import os, random, itertools, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report, confusion_matrix)
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm

from config import (
    SEED, EPOCHS, PATIENCE, FREEZE_EPOCHS, GRAD_ACCUM,
    LR_EVASION, LR_STANCE, WARMUP_RATIO, OUTPUT_DIR,
    FULL_MODEL_PATH, EVASION_LABELS, STANCE_LABELS,
)
from data import load_all_data
from model import MultiTaskV3C


# ════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ════════════════════════════════════════════════════════════════════════════

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")


# ════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ════════════════════════════════════════════════════════════════════════════

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
        if not (torch.isnan(loss) or torch.isinf(loss)):
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
    print(m["cm"].to_string()); print("=" * 60)


def get_pe_f1(report_str):
    for line in report_str.split("\n"):
        if "Partially Evasive" in line:
            try: return float(line.split()[3])
            except: return 0.0
    return 0.0


# ════════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════════

def train(model, loaders, save_path):
    best_avg_f1 = 0.0; patience_ctr = 0

    shared  = (list(model.encoder.parameters()) +
               list(model.token_cross_attn.parameters()))
    ev_task = (list(model.evasion_head.parameters()) +
               list(model.evasion_gate.parameters()))
    st_task = (list(model.stance_head.parameters()) +
               list(model.stance_gate.parameters()))

    optimizer = AdamW([
        {"params": ev_task, "lr": LR_EVASION},
        {"params": st_task, "lr": LR_STANCE},
        {"params": shared,  "lr": LR_EVASION},
    ], weight_decay=0.01)

    # 3:1 evasion priority → 4 batches per stance batch
    n_steps   = len(loaders["stance_train"]) * 4 * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * n_steps), n_steps)

    # Freeze encoder for first FREEZE_EPOCHS
    for p in model.encoder.parameters(): p.requires_grad = False

    print(f"  Params:{sum(p.numel() for p in model.parameters()):,}  "
          f"Sampling:3:1  Scheduler:Cosine  Patience:{PATIENCE}")
    print(f"  Freeze:{FREEZE_EPOCHS} epochs  LR_e:{LR_EVASION}  LR_s:{LR_STANCE}")
    print(f"  Saving best checkpoint to: {save_path}")

    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            for p in model.encoder.parameters(): p.requires_grad = True
            print(f"  Epoch {epoch}: BERT UNFROZEN.")

        model.train()
        e_cycle  = itertools.cycle(list(loaders["evasion_train"]))
        combined = []
        for sb in loaders["stance_train"]:
            combined.append(next(e_cycle))
            combined.append(next(e_cycle))
            combined.append(next(e_cycle))
            combined.append(sb)   # 3:1 evasion priority
        total = len(combined)
        run_e = run_s = ne = ns = 0; t0 = time.time()
        pbar  = tqdm(combined, total=total, desc=f"Ep {epoch}/{EPOCHS}", leave=True)

        for step, batch in enumerate(pbar, 1):
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
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(); continue
            (loss / GRAD_ACCUM).backward()
            if task == 0: run_e += loss.item(); ne += 1
            else:         run_s += loss.item(); ns += 1
            if step % GRAD_ACCUM == 0 or step == total:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if step % 100 == 0 or step == total:
                avg_e = run_e / ne if ne else 0
                avg_s = run_s / ns if ns else 0
                pbar.set_description(f"Ep {epoch} | E:{avg_e:.3f} S:{avg_s:.3f}")
        pbar.close()

        elapsed = (time.time() - t0) / 60
        ev = evaluate(model, loaders["evasion_val"], EVASION_LABELS)
        sv = evaluate(model, loaders["stance_val"],  STANCE_LABELS)
        avg = (ev["f1"] + sv["f1"]) / 2
        pe  = get_pe_f1(ev["report"])
        is_best = avg > best_avg_f1
        print(f"  Ep{epoch} ({elapsed:.1f}m) E:{run_e/ne:.4f} S:{run_s/ns:.4f} | "
              f"Val E:{ev['f1']:.4f} S:{sv['f1']:.4f} Avg:{avg:.4f} PE:{pe:.2f} "
              f"{'BEST' if is_best else f'p:{patience_ctr+1}/{PATIENCE}'}")
        print("  Val per-class evasion:")
        for line in ev["report"].strip().split("\n"):
            if any(lbl in line for lbl in EVASION_LABELS):
                print(f"    {line.strip()}")

        if is_best:
            best_avg_f1 = avg; patience_ctr = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("  Early stopping."); break
        print()

    # Reload best and run final test
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.unfreeze_all()
    et = evaluate(model, loaders["evasion_test"], EVASION_LABELS)
    st = evaluate(model, loaders["stance_test"],  STANCE_LABELS)
    print(f"  TEST: E:{et['f1']:.4f} S:{st['f1']:.4f} "
          f"Avg:{(et['f1']+st['f1'])/2:.4f} PE:{get_pe_f1(et['report']):.2f} "
          f"Acc:{et['acc']*100:.2f}%")
    return et, st


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    loaders, raw, (ew, sw) = load_all_data()
    Qete, Aete, yete, _, _, _, Qste, Aste, yste = raw

    print("\n" + "=" * 65)
    print("  TRAINING: V3-C + TokenCrossAttn + FocalLoss + ContrastiveLoss")
    print("=" * 65)

    model = MultiTaskV3C(ew.to(device), sw.to(device)).to(device)
    e_test, s_test = train(model, loaders, FULL_MODEL_PATH)

    print_eval(e_test, "EVASION TEST — V3-C Final")
    print_eval(s_test, "STANCE  TEST — V3-C Final")

    # Save predictions
    pd.DataFrame({
        "question": Qete,
        "answer":   Aete,
        "true": [EVASION_LABELS[l] for l in e_test["labels"]],
        "pred": [EVASION_LABELS[p] for p in e_test["preds"]],
    }).to_csv(os.path.join(OUTPUT_DIR, "preds_v3c_final_evasion.csv"), index=False)

    pd.DataFrame({
        "target": Qste,
        "tweet":  Aste,
        "true": [STANCE_LABELS[l] for l in s_test["labels"]],
        "pred": [STANCE_LABELS[p] for p in s_test["preds"]],
    }).to_csv(os.path.join(OUTPUT_DIR, "preds_v3c_final_stance.csv"), index=False)

    sz = os.path.getsize(FULL_MODEL_PATH) / (1024 * 1024)
    print(f"\nModel saved: {FULL_MODEL_PATH}  ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
