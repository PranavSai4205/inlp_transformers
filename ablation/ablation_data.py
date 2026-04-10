import os, re, random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertTokenizer

from ablation_config import (
    BERT_MODEL, SEED, MAX_LEN, STANCE_MAX_LEN, Q_MAX_LEN,
    MAX_CLAUSES, CLAUSE_LEN, BATCH_SIZE,
    EVASION_LABELS, EVASION_LABEL2ID, STANCE_LABELS, STANCE_LABEL2ID,
    EVASION_CSV, STANCE_TRAIN_CSVS, STANCE_VAL_CSVS, STANCE_TEST_CSVS,
)

# ── Tokenizer ─────────────────────────────────────────────────────────────
tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL)

# ══════════════════════════════════════════════════════════════════════════
# Augmentation  (3-op EDA — simpler than final model's 5-strategy pipeline)
# ══════════════════════════════════════════════════════════════════════════

SYNONYM_MAP = {
    "question": ["query", "inquiry"], "answer": ["response", "reply"],
    "policy": ["plan", "strategy"], "government": ["administration", "authority"],
    "issue": ["matter", "concern", "problem"], "important": ["crucial", "significant", "vital"],
    "believe": ["think", "consider", "feel"], "people": ["citizens", "individuals", "public"],
    "country": ["nation", "state"], "support": ["back", "endorse", "advocate"],
    "change": ["shift", "reform", "alter"], "need": ["require", "must"],
    "work": ["function", "operate", "effort"], "make": ["create", "produce", "form"],
    "good": ["beneficial", "positive", "effective"], "said": ["stated", "noted", "mentioned"],
    "know": ["understand", "recognize", "realize"], "think": ["believe", "consider", "feel"],
    "want": ["desire", "seek", "aim"], "time": ["period", "moment", "point"],
    "new": ["recent", "fresh", "updated"], "political": ["governmental", "civic"],
    "years": ["decades", "period"], "major": ["significant", "key", "primary"],
    "different": ["various", "distinct", "alternative"],
}


def eda_synonym_replace(text, n=3, seed=None):
    if seed is not None: random.seed(seed)
    words = text.split()
    replaceable = [(i, w.lower().strip('.,!?;:')) for i, w in enumerate(words)
                   if w.lower().strip('.,!?;:') in SYNONYM_MAP]
    if not replaceable: return text
    chosen = random.sample(replaceable, min(n, len(replaceable)))
    new_words = words[:]
    for i, w in chosen:
        syn = random.choice(SYNONYM_MAP[w])
        if words[i][0].isupper(): syn = syn.capitalize()
        new_words[i] = syn
    return " ".join(new_words)


def eda_random_swap(text, n=2, seed=None):
    if seed is not None: random.seed(seed)
    words = text.split()
    if len(words) < 4: return text
    new_words = words[:]
    for _ in range(n):
        i, j = random.sample(range(len(new_words)), 2)
        new_words[i], new_words[j] = new_words[j], new_words[i]
    return " ".join(new_words)


def eda_random_deletion(text, p=0.1, seed=None):
    if seed is not None: random.seed(seed)
    words = text.split()
    if len(words) <= 4: return text
    new_words = [w for w in words if random.random() > p]
    return " ".join(new_words) if new_words else text


def augment_pe_samples(questions, answers, labels,
                       target_label_id=1, augment_factor=8, seed=42):
    """
    Augment the Partially Evasive minority class using 4 EDA operations
    (cycled across augment_factor copies per sample).
    Seed offset: seed + i*100 + k  (matches ablation notebook exactly).
    """
    aug_q, aug_a, aug_l = [], [], []
    pe_indices = [i for i, l in enumerate(labels) if l == target_label_id]
    print(f"  Augmenting {len(pe_indices)} PE samples (factor={augment_factor})...")

    ops = [
        lambda a, s: eda_synonym_replace(a, n=3, seed=s),
        lambda a, s: eda_random_swap(a, n=2, seed=s),
        lambda a, s: eda_random_deletion(a, p=0.1, seed=s),
        lambda a, s: eda_synonym_replace(eda_random_swap(a, n=1, seed=s), n=2, seed=s + 1),
    ]

    for i, idx in enumerate(pe_indices):
        q, a, l = questions[idx], answers[idx], labels[idx]
        for k in range(augment_factor):
            op = ops[k % len(ops)]
            aug_q.append(q)
            aug_a.append(op(a, seed + i * 100 + k))   # positional arg, matches notebook
            aug_l.append(l)

    print(f"  Generated {len(aug_l)} augmented PE samples.")
    return aug_q, aug_a, aug_l


# ══════════════════════════════════════════════════════════════════════════
# Label mapping  (numeric prefix format — different from final model)
# ══════════════════════════════════════════════════════════════════════════

def map_evasion_label(raw):
    """
    Maps raw CSV label strings to 3 coarse evasion classes.
      Non-Evasive      <- "Explicit", "Implicit"
      Partially Evasive <- "Partial/half-answer", "Clarification"
      Evasive          <- "Dodging", "General", "Deflection",
                          "Declining to answer", "Claims ignorance"
    """
    raw = str(raw).strip()
    rl  = raw.lower()
    if rl in ("explicit", "implicit"):
        return "Non-Evasive"
    if rl in ("partial/half-answer", "clarification") or "partial" in rl or "half" in rl:
        return "Partially Evasive"
    return "Evasive"

# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════

def load_evasion(filepath):
    print("Loading QA Evasion...")
    df = pd.read_csv(filepath)
    df["question"] = df["interview_question"].fillna("").str.strip()
    df["answer"]   = df["interview_answer"].fillna("").str.strip()
    df["coarse_label"] = df["label"].apply(map_evasion_label)
    df["label_id"]     = df["coarse_label"].map(EVASION_LABEL2ID).astype(int)
    df = df.dropna(subset=["question", "answer", "label_id"]).reset_index(drop=True)

    for lbl in EVASION_LABELS:
        n = (df["coarse_label"] == lbl).sum()
        print(f"  {lbl:22s}: {n:4d} ({n / len(df) * 100:.1f}%)")

    idx = list(range(len(df))); lbl = df["label_id"].tolist()
    itr, itmp, _, ytmp = train_test_split(idx, lbl, test_size=0.20,
                                          random_state=SEED, stratify=lbl)
    iva, ite, _, _     = train_test_split(itmp, ytmp, test_size=0.50,
                                          random_state=SEED, stratify=ytmp)
    q, a, l = df["question"].tolist(), df["answer"].tolist(), df["label_id"].tolist()
    def ex(ii): return [q[i] for i in ii], [a[i] for i in ii], [l[i] for i in ii]
    tr, va, te = ex(itr), ex(iva), ex(ite)
    print(f"  Train:{len(itr)}  Val:{len(iva)}  Test:{len(ite)}")
    return tr, va, te


def load_stance_split(filepaths, split_name=""):
    dfs = []
    for fp in filepaths:
        if os.path.exists(fp): dfs.append(pd.read_csv(fp))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["Stance"].isin(STANCE_LABEL2ID)].reset_index(drop=True)
    df["label_id"] = df["Stance"].map(STANCE_LABEL2ID).astype(int)
    df["target"]   = df["Target"].fillna("").str.strip()
    df["tweet"]    = df["Tweet"].fillna("").str.strip()
    df = df.dropna(subset=["target", "tweet", "label_id"]).reset_index(drop=True)
    return df["target"].tolist(), df["tweet"].tolist(), df["label_id"].tolist()


# ══════════════════════════════════════════════════════════════════════════
# Clause splitting
# ══════════════════════════════════════════════════════════════════════════

def split_into_clauses(text):
    """Split answer into up to MAX_CLAUSES clauses. Returns at least 1."""
    parts = re.split(
        r'(?<=[.!?])\s+|(?:\s+(?:but|however|although|whereas|while|yet|'
        r'because|since|so|therefore|nevertheless|nonetheless)\s+)', text)
    parts = [p.strip() for p in parts if len(p.strip()) > 5]
    return parts[:MAX_CLAUSES] if parts else [text[:500]]


# ══════════════════════════════════════════════════════════════════════════
# Dataset class
# ══════════════════════════════════════════════════════════════════════════

class TaskDataset(Dataset):
    """
    task_id = 0 → evasion  (ta=question, tb=answer)
    task_id = 1 → stance   (ta=target,   tb=tweet)
    """
    def __init__(self, ta, tb, labels, max_len, task_id):
        self.ta, self.tb, self.labels = ta, tb, labels
        self.max_len, self.task_id    = max_len, task_id

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        enc  = tokenizer(self.ta[idx], self.tb[idx], truncation=True,
                         padding="max_length", max_length=self.max_len,
                         return_tensors="pt")
        qenc = tokenizer(self.ta[idx], truncation=True,
                         padding="max_length", max_length=Q_MAX_LEN,
                         return_tensors="pt")
        clauses = split_into_clauses(self.tb[idx]) if self.task_id == 0 else [self.tb[idx]]
        ids_l, masks_l = [], []
        for c in clauses[:MAX_CLAUSES]:
            ce = tokenizer(self.ta[idx], c, truncation=True,
                           padding="max_length", max_length=CLAUSE_LEN,
                           return_tensors="pt")
            ids_l.append(ce["input_ids"].squeeze(0))
            masks_l.append(ce["attention_mask"].squeeze(0))
        n_real = len(ids_l)
        while len(ids_l) < MAX_CLAUSES:
            ids_l.append(torch.zeros(CLAUSE_LEN, dtype=torch.long))
            masks_l.append(torch.zeros(CLAUSE_LEN, dtype=torch.long))
        return {
            "input_ids":        enc["input_ids"].squeeze(0),
            "attention_mask":   enc["attention_mask"].squeeze(0),
            "q_input_ids":      qenc["input_ids"].squeeze(0),
            "q_attention_mask": qenc["attention_mask"].squeeze(0),
            "clause_ids":       torch.stack(ids_l),
            "clause_masks":     torch.stack(masks_l),
            "n_clauses":        torch.tensor(n_real, dtype=torch.long),
            "labels":           torch.tensor(self.labels[idx], dtype=torch.long),
            "task_id":          torch.tensor(self.task_id, dtype=torch.long),
        }


def make_loader(ta, tb, y, tid, ml, shuffle):
    return DataLoader(TaskDataset(ta, tb, y, ml, tid),
                      batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


# ══════════════════════════════════════════════════════════════════════════
# Top-level loader
# ══════════════════════════════════════════════════════════════════════════

def load_all_data():
    """
    Load, augment, and return all splits + DataLoaders + class weights.

    Returns:
      loaders : dict of DataLoader
      raw     : (Qete, Aete, yete, Qste, Aste, yste)
      weights : (ew, sw) — class-weight tensors on CPU
    """
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("=" * 60)
    (Qetr, Aetr, yetr), (Qeva, Aeva, yeva), (Qete, Aete, yete) = load_evasion(EVASION_CSV)

    print("\nApplying EDA augmentation to training set (PE class)...")
    aug_q, aug_a, aug_l = augment_pe_samples(Qetr, Aetr, yetr,
                                              target_label_id=1, augment_factor=8)
    Qetr_aug = Qetr + aug_q; Aetr_aug = Aetr + aug_a; yetr_aug = yetr + aug_l
    combined = list(zip(Qetr_aug, Aetr_aug, yetr_aug))
    random.seed(SEED); random.shuffle(combined)
    Qetr_aug, Aetr_aug, yetr_aug = [list(x) for x in zip(*combined)]
    print(f"  Augmented evasion train: {len(yetr_aug)} samples")
    for lid, lbl in enumerate(EVASION_LABELS):
        n = sum(1 for y in yetr_aug if y == lid)
        print(f"  {lbl:22s}: {n:4d} ({n / len(yetr_aug) * 100:.1f}%)")

    Qstr, Astr, ystr = load_stance_split(STANCE_TRAIN_CSVS, "train")
    Qsva, Asva, ysva = load_stance_split(STANCE_VAL_CSVS,   "val")
    Qste, Aste, yste = load_stance_split(STANCE_TEST_CSVS,  "test")
    print("=" * 60)
    print(f"Stance train:{len(ystr)}  val:{len(ysva)}  test:{len(yste)}")

    ew = torch.tensor(compute_class_weight("balanced",
                       classes=np.array([0, 1, 2]), y=yetr_aug), dtype=torch.float)
    sw = torch.tensor(compute_class_weight("balanced",
                       classes=np.array([0, 1]),    y=ystr),     dtype=torch.float)
    print(f"Evasion class weights (post-aug): {[round(x, 4) for x in ew.tolist()]}")
    print(f"Stance  class weights: {[round(x, 4) for x in sw.tolist()]}")

    loaders = {
        "evasion_train": make_loader(Qetr_aug, Aetr_aug, yetr_aug, 0, MAX_LEN, True),
        "evasion_val":   make_loader(Qeva,     Aeva,     yeva,     0, MAX_LEN, False),
        "evasion_test":  make_loader(Qete,     Aete,     yete,     0, MAX_LEN, False),
        "stance_train":  make_loader(Qstr,     Astr,     ystr,     1, STANCE_MAX_LEN, True),
        "stance_val":    make_loader(Qsva,     Asva,     ysva,     1, STANCE_MAX_LEN, False),
        "stance_test":   make_loader(Qste,     Aste,     yste,     1, STANCE_MAX_LEN, False),
    }
    for k, v in loaders.items():
        print(f"  {k:<20} {len(v.dataset):>6,} samples  {len(v):>5,} batches")
    print("DataLoaders ready.")

    raw = (Qete, Aete, yete, Qste, Aste, yste)
    return loaders, raw, (ew, sw)
