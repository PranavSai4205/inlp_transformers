import os

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR   = "./dataset"     # folder containing QAEvasion.csv and P-Stance CSVs
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────
BERT_MODEL = "distilbert-base-uncased"

# ── Sequence lengths ──────────────────────────────────────────────────────
MAX_LEN        = 256
STANCE_MAX_LEN = 128
Q_MAX_LEN      = 64
MAX_CLAUSES    = 6     # NOTE: 6, not 4
CLAUSE_LEN     = 64    # NOTE: 64, not 96

# ── Training ──────────────────────────────────────────────────────────────
BATCH_SIZE    = 16
GRAD_ACCUM    = 2       # effective batch = 32
EPOCHS        = 5
DROPOUT       = 0.3
WARMUP_RATIO  = 0.1
FREEZE_EPOCHS = 2       # NOTE: 2, not 3
LR_EVASION    = 1e-5    # NOTE: 1e-5, not 5e-6
LR_STANCE     = 2e-5    # NOTE: 2e-5, not 1e-5

# ── Labels ────────────────────────────────────────────────────────────────
EVASION_LABELS   = ["Non-Evasive", "Partially Evasive", "Evasive"]
EVASION_LABEL2ID = {l: i for i, l in enumerate(EVASION_LABELS)}
STANCE_LABELS    = ["FAVOR", "AGAINST"]
STANCE_LABEL2ID  = {l: i for i, l in enumerate(STANCE_LABELS)}

# ── Data files ────────────────────────────────────────────────────────────
# NOTE: older dataset file is QAEvasion.csv (not QEvasion_train.csv)
EVASION_CSV       = os.path.join(DATA_DIR, "QEvasion_train.csv")
STANCE_TRAIN_CSVS = [os.path.join(DATA_DIR, f) for f in
                     ["raw_train_biden.csv", "raw_train_trump.csv", "raw_train_bernie.csv"]]
STANCE_VAL_CSVS   = [os.path.join(DATA_DIR, f) for f in
                     ["raw_val_biden.csv",   "raw_val_trump.csv",   "raw_val_bernie.csv"]]
STANCE_TEST_CSVS  = [os.path.join(DATA_DIR, f) for f in
                     ["raw_test_biden.csv",  "raw_test_trump.csv",  "raw_test_bernie.csv"]]

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

# ── Ablation variants ─────────────────────────────────────────────────────
# Each entry defines one ablation run.
# use_gating / use_cross_attn / use_graph / use_clause_pool are toggled.
ABLATION_CONFIGS = [
    {
        "name":  "V3-A: No Graph Reasoning",
        "tag":   "v3_ablation_no_graph",
        "flags": dict(use_gating=True, use_cross_attn=True,
                      use_graph=False, use_clause_pool=True),
        "desc":  "Removes GraphReasoningLayer. Keeps: Gating + CrossAttn + ClausePool.",
    },
    {
        "name":  "V3-B: No Cross-Attention",
        "tag":   "v3_ablation_no_crossattn",
        "flags": dict(use_gating=True, use_cross_attn=False,
                      use_graph=True,  use_clause_pool=True),
        "desc":  "Removes CrossAttention. Keeps: Gating + Graph + ClausePool.",
    },
    {
        "name":  "V3-C: No Clause Hierarchy (Gating only)",
        "tag":   "v3_ablation_no_clause",
        "flags": dict(use_gating=True, use_cross_attn=False,
                      use_graph=False, use_clause_pool=False),
        "desc":  "Removes all clause components. Keeps: Gating only (= V2 inside V3 framework).",
    },
    {
        "name":  "V3-D: No Gating",
        "tag":   "v3_ablation_no_gating",
        "flags": dict(use_gating=False, use_cross_attn=True,
                      use_graph=True,   use_clause_pool=True),
        "desc":  "Removes TokenLevelGating. Keeps: CrossAttn + Graph + ClausePool.",
    },
]
