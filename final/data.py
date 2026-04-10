import os, re, random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertTokenizer

from config import (
    BERT_MODEL, SEED, MAX_LEN, STANCE_MAX_LEN, Q_MAX_LEN,
    MAX_CLAUSES, CLAUSE_LEN, BATCH_SIZE,
    EVASION_LABELS, EVASION_LABEL2ID, STANCE_LABELS, STANCE_LABEL2ID,
    EVASION_CSV, STANCE_TRAIN_CSVS, STANCE_VAL_CSVS, STANCE_TEST_CSVS,
)

# ── Tokenizer (loaded once, shared everywhere) ───────────────────────────────
tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL)

# ════════════════════════════════════════════════════════════════════════════
# Augmentation
# ════════════════════════════════════════════════════════════════════════════

SYNONYM_MAP = {
    "question":["query","inquiry"],"answer":["response","reply"],
    "policy":["plan","strategy"],"government":["administration","authority"],
    "issue":["matter","concern","problem"],"important":["crucial","significant","vital"],
    "believe":["think","consider","feel"],"people":["citizens","individuals","public"],
    "country":["nation","state"],"support":["back","endorse","advocate"],
    "change":["shift","reform","alter"],"need":["require","must"],
    "work":["function","operate","effort"],"make":["create","produce","form"],
    "good":["beneficial","positive","effective"],"said":["stated","noted","mentioned"],
    "know":["understand","recognize","realize"],"think":["believe","consider","feel"],
    "want":["desire","seek","aim"],"time":["period","moment","point"],
    "new":["recent","fresh","updated"],"political":["governmental","civic"],
    "major":["significant","key","primary"],"different":["various","distinct","alternative"],
    "president":["commander-in-chief","leader","executive"],
    "congress":["legislature","lawmakers","senate"],
    "economy":["market","financial system","fiscal situation"],
    "certainly":["absolutely","definitely","of course"],
    "perhaps":["possibly","maybe","one could argue"],
    "however":["that said","nevertheless","on the other hand"],
    "basically":["fundamentally","essentially","at its core"],
}

EVASION_PHRASES = {
    "i think we should":"it is my view that we ought to",
    "i believe that":"it is my conviction that",
    "we need to":"it is essential that we",
    "i want to":"my intention is to",
    "we are working on":"efforts are underway to",
    "i don't think":"it is not my view that",
    "the fact is":"what we know is",
    "i would say":"my position would be",
    "let me be clear":"to make this absolutely clear",
    "the truth is":"the reality of the situation is",
    "we have to":"it is necessary that we",
    "i am committed to":"my commitment remains to",
    "going forward":"in the coming period",
    "at the end of the day":"ultimately",
    "we must":"it is imperative that we",
    "i am confident":"i have full confidence",
}

STOPWORDS = {"the","a","an","is","are","was","were","be","been","being",
             "have","has","had","do","does","did","will","would","could",
             "should","may","might","to","of","in","for","on","with",
             "at","by","from","as","into","i","we","it","this","that"}


def aug_synonym_replace(text, n=3, seed=None):
    if seed is not None: random.seed(seed)
    words = text.split()
    rep = [(i, w.lower().strip('.,!?;:\"\'()')) for i, w in enumerate(words)
           if w.lower().strip('.,!?;:\"\'()') in SYNONYM_MAP]
    if not rep: return text
    chosen = random.sample(rep, min(n, len(rep)))
    nw = words[:]
    for i, w in chosen:
        syn = random.choice(SYNONYM_MAP[w])
        if words[i][0].isupper(): syn = syn[0].upper() + syn[1:]
        nw[i] = syn
    return " ".join(nw)


def aug_structure_swap(text, n=2, seed=None):
    if seed is not None: random.seed(seed)
    clauses = re.split(r'(?<=[.!?,;])\s+', text)
    clauses = [c for c in clauses if len(c.split()) >= 4]
    if not clauses:
        words = text.split()
        if len(words) < 4: return text
        for _ in range(n):
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        return " ".join(words)
    nc = []
    for clause in clauses:
        words = clause.split()
        if len(words) >= 4 and random.random() < 0.6:
            for _ in range(min(n, len(words) // 2)):
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
        nc.append(" ".join(words))
    return " ".join(nc)


def aug_conservative_deletion(text, p=0.07, seed=None):
    if seed is not None: random.seed(seed)
    words = text.split()
    if len(words) <= 6: return text
    nw = []
    for w in words:
        clean = w.lower().strip('.,!?;:\"\'()')
        if clean in STOPWORDS or len(clean) <= 2: nw.append(w)
        elif random.random() > p: nw.append(w)
    return " ".join(nw) if len(nw) >= len(words) // 2 else text


def aug_phrase_paraphrase(text, seed=None):
    if seed is not None: random.seed(seed)
    tl = text.lower()
    cands = [(p, r) for p, r in EVASION_PHRASES.items() if p in tl]
    if not cands: return aug_synonym_replace(text, n=2, seed=seed)
    chosen = random.sample(cands, min(random.randint(1, 2), len(cands)))
    result = text
    for phrase, replacement in chosen:
        idx = result.lower().find(phrase)
        if idx >= 0:
            if idx == 0 or result[idx - 1] in '.!?':
                replacement = replacement[0].upper() + replacement[1:]
            result = result[:idx] + replacement + result[idx + len(phrase):]
    return result


def aug_combined_chain(text, seed=None):
    if seed is not None: random.seed(seed)
    return aug_structure_swap(aug_synonym_replace(text, n=2, seed=seed), n=1, seed=seed + 1)


AUG_OPS = [aug_synonym_replace, aug_structure_swap, aug_conservative_deletion,
           aug_phrase_paraphrase, aug_combined_chain]


def augment_pe_samples(questions, answers, labels, target_label_id=1,
                       augment_factor=8, seed=42):
    aug_q, aug_a, aug_l = [], [], []
    pe_idx = [i for i, l in enumerate(labels) if l == target_label_id]
    print(f"  PE samples:{len(pe_idx)}  factor:{augment_factor}x  -> +{len(pe_idx) * augment_factor} samples")
    for i, idx in enumerate(pe_idx):
        q, a, l = questions[idx], answers[idx], labels[idx]
        for k in range(augment_factor):
            op = AUG_OPS[k % len(AUG_OPS)]
            aug_q.append(q); aug_a.append(op(a, seed + i * 500 + k)); aug_l.append(l)
    return aug_q, aug_a, aug_l


# ════════════════════════════════════════════════════════════════════════════
# Label mapping
# ════════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════

def load_evasion(filepath):
    print("Loading QA Evasion...")
    df = pd.read_csv(filepath)
    df["question"] = df["interview_question"].fillna("").str.strip()
    df["answer"]   = df["interview_answer"].fillna("").str.strip()

    print(f"  Unique raw labels ({df['label'].nunique()}):")
    for v, cnt in df["label"].value_counts().items():
        mapped = map_evasion_label(v)
        print(f"    {str(v):<45} -> {mapped} (n={cnt})")

    df["coarse_label"] = df["label"].apply(map_evasion_label)
    df["label_id"]     = df["coarse_label"].map(EVASION_LABEL2ID).astype(int)
    df = df.dropna(subset=["question", "answer", "label_id"]).reset_index(drop=True)
    print(f"  Total:{len(df)}")
    for lbl in EVASION_LABELS:
        n = (df["coarse_label"] == lbl).sum()
        print(f"  {lbl:22s}: {n:4d} ({n / len(df) * 100:.1f}%)")

    present = df["label_id"].unique()
    for lid, lbl in enumerate(EVASION_LABELS):
        if lid not in present:
            raise ValueError(
                f"Class '{lbl}' (id={lid}) has 0 samples after mapping.\n"
                f"Check the raw label values printed above and fix map_evasion_label().")

    idx = list(range(len(df))); lbl = df["label_id"].tolist()
    itr, itmp, _, ytmp = train_test_split(idx, lbl, test_size=0.20, random_state=SEED, stratify=lbl)
    iva, ite, _, _     = train_test_split(itmp, ytmp, test_size=0.50, random_state=SEED, stratify=ytmp)
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


# ════════════════════════════════════════════════════════════════════════════
# Clause splitting
# ════════════════════════════════════════════════════════════════════════════

def split_into_clauses(text):
    """
    Split answer into up to MAX_CLAUSES clauses.
    Always returns at least 1 clause; filters fragments ≤5 chars.
    """
    text = text.strip() or "empty"
    parts = re.split(
        r'(?<=[.!?])\s+|(?:\s+(?:but|however|although|whereas|while|yet|'
        r'because|since|so|therefore|nevertheless|nonetheless)\s+)', text)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]
    if not parts:
        parts = [text[:CLAUSE_LEN * 2]]
    return parts[:MAX_CLAUSES]


# ════════════════════════════════════════════════════════════════════════════
# Dataset class
# ════════════════════════════════════════════════════════════════════════════

class TaskDataset(Dataset):
    """
    task_id = 0 → evasion  (ta=question, tb=answer)
    task_id = 1 → stance   (ta=target, tb=tweet)
    """
    def __init__(self, ta, tb, labels, max_len, task_id):
        self.ta, self.tb, self.labels = ta, tb, labels
        self.max_len, self.task_id    = max_len, task_id

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        enc  = tokenizer(self.ta[idx], self.tb[idx], truncation=True,
                         padding="max_length", max_length=self.max_len, return_tensors="pt")
        qenc = tokenizer(self.ta[idx], truncation=True,
                         padding="max_length", max_length=Q_MAX_LEN, return_tensors="pt")
        clauses = split_into_clauses(self.tb[idx]) if self.task_id == 0 else [self.tb[idx]]
        ids_l, masks_l = [], []
        for c in clauses[:MAX_CLAUSES]:
            ce = tokenizer(self.ta[idx], c, truncation=True,
                           padding="max_length", max_length=CLAUSE_LEN, return_tensors="pt")
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


# ════════════════════════════════════════════════════════════════════════════
# Top-level load function
# ════════════════════════════════════════════════════════════════════════════

def load_all_data():
    """
    Load + augment all splits, return:
      loaders  : dict of DataLoader
      raw data : (Qete, Aete, yete, Qeva, Aeva, yeva, Qste, Aste, yste)
      weights  : (ew, sw)  class-weight tensors on CPU
    """
    import torch
    random.seed(SEED); import numpy as np; np.random.seed(SEED); torch.manual_seed(SEED)

    print("=" * 60)
    (Qetr, Aetr, yetr), (Qeva, Aeva, yeva), (Qete, Aete, yete) = load_evasion(EVASION_CSV)

    print("\nAugmenting PE class in training set (8x)...")
    aug_q, aug_a, aug_l = augment_pe_samples(Qetr, Aetr, yetr, target_label_id=1, augment_factor=8)
    Qetr_aug = Qetr + aug_q; Aetr_aug = Aetr + aug_a; yetr_aug = yetr + aug_l
    combined = list(zip(Qetr_aug, Aetr_aug, yetr_aug))
    random.seed(SEED); random.shuffle(combined)
    Qetr_aug, Aetr_aug, yetr_aug = [list(x) for x in zip(*combined)]
    print(f"  Augmented train:{len(yetr_aug)}")
    for lid, lbl in enumerate(EVASION_LABELS):
        n = sum(1 for y in yetr_aug if y == lid)
        print(f"  {lbl:22s}: {n:4d} ({n / len(yetr_aug) * 100:.1f}%)")

    print()
    Qstr, Astr, ystr = load_stance_split(STANCE_TRAIN_CSVS, "train")
    Qsva, Asva, ysva = load_stance_split(STANCE_VAL_CSVS,   "val")
    Qste, Aste, yste = load_stance_split(STANCE_TEST_CSVS,  "test")
    print(f"Stance -> train:{len(ystr)}  val:{len(ysva)}  test:{len(yste)}")
    print("=" * 60)

    ew = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=yetr_aug), dtype=torch.float)
    sw = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1]),    y=ystr),     dtype=torch.float)
    print(f"Evasion weights: {[round(x, 4) for x in ew.tolist()]}")
    print(f"Stance  weights: {[round(x, 4) for x in sw.tolist()]}")

    loaders = {
        "evasion_train": make_loader(Qetr_aug, Aetr_aug, yetr_aug, 0, MAX_LEN, True),
        "evasion_val":   make_loader(Qeva,     Aeva,     yeva,     0, MAX_LEN, False),
        "evasion_test":  make_loader(Qete,     Aete,     yete,     0, MAX_LEN, False),
        "stance_train":  make_loader(Qstr,     Astr,     ystr,     1, STANCE_MAX_LEN, True),
        "stance_val":    make_loader(Qsva,     Asva,     ysva,     1, STANCE_MAX_LEN, False),
        "stance_test":   make_loader(Qste,     Aste,     yste,     1, STANCE_MAX_LEN, False),
    }
    for k, v in loaders.items():
        print(f"  {k:<20} {len(v.dataset):>7,} samples  {len(v):>4} batches")

    raw = (Qete, Aete, yete, Qeva, Aeva, yeva, Qste, Aste, yste)
    return loaders, raw, (ew, sw)
