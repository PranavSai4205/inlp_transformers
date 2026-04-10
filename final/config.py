import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = "./dataset"
OUTPUT_DIR = "./output"           # where checkpoints and predictions are saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model ────────────────────────────────────────────────────────────────────
BERT_MODEL = "distilbert-base-uncased"

# ── Sequence lengths ─────────────────────────────────────────────────────────
MAX_LEN        = 256   # full [Q;A] for evasion
STANCE_MAX_LEN = 128   # full [Target;Tweet] for stance
Q_MAX_LEN      = 64    # question-only encoding window
MAX_CLAUSES    = 4
CLAUSE_LEN     = 96

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE    = 16
GRAD_ACCUM    = 2          # effective batch = 32
EPOCHS        = 7
PATIENCE      = 3
DROPOUT       = 0.3
WARMUP_RATIO  = 0.1
FREEZE_EPOCHS = 3          # encoder frozen for first 3 epochs
LR_EVASION    = 5e-6
LR_STANCE     = 1e-5
LABEL_SMOOTH  = 0.1        # stance CE only; focal loss does NOT use smoothing

# ── Loss ─────────────────────────────────────────────────────────────────────
FOCAL_GAMMA      = 3.0     # Focal Loss gamma (higher = more focus on hard/minority examples)
CONTRASTIVE_TAU  = 0.1     # NT-Xent temperature
CONTRASTIVE_W    = 0.10    # weight of contrastive term added to classification loss

# ── Labels ───────────────────────────────────────────────────────────────────
EVASION_LABELS   = ["Non-Evasive", "Partially Evasive", "Evasive"]
EVASION_LABEL2ID = {l: i for i, l in enumerate(EVASION_LABELS)}
STANCE_LABELS    = ["FAVOR", "AGAINST"]
STANCE_LABEL2ID  = {l: i for i, l in enumerate(STANCE_LABELS)}

# ── Data files ───────────────────────────────────────────────────────────────
EVASION_CSV       = os.path.join(DATA_DIR, "QEvasion_train.csv")
STANCE_TRAIN_CSVS = [os.path.join(DATA_DIR, f) for f in
                     ["raw_train_biden.csv", "raw_train_trump.csv", "raw_train_bernie.csv"]]
STANCE_VAL_CSVS   = [os.path.join(DATA_DIR, f) for f in
                     ["raw_val_biden.csv",   "raw_val_trump.csv",   "raw_val_bernie.csv"]]
STANCE_TEST_CSVS  = [os.path.join(DATA_DIR, f) for f in
                     ["raw_test_biden.csv",  "raw_test_trump.csv",  "raw_test_bernie.csv"]]

# ── Checkpoint ───────────────────────────────────────────────────────────────
FULL_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_v3c_final.pt")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
