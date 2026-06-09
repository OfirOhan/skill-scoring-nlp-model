"""
dataset.py – Data loading, preprocessing, and train/val/test splitting
for the CORAL ordinal-regression scoring model.

The split is performed on the DataFrame (not on plain lists) so that each row's
original CSV index is preserved. evaluate.py relies on the test frame's index to
join model predictions back to per-row retrieval provenance in
retrieval_meta.jsonl (both are keyed by that same CSV row position).
"""

import re

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

import config


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class ScoringDataset(Dataset):
    """
    Reads rows with columns:  skill, chunk1, chunk2, chunk3, label
    • Concatenates input as:  "{skill} {sep} {chunk1} {chunk2} {chunk3}"
      where {sep} is the backbone's separator token (DeBERTa: '[SEP]')
    • Tokenises with the model's AutoTokenizer (pad / truncate to MAX_LEN)
    • Shifts labels from 1-5 → 0-4 (zero-indexed for CORAL)
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),           # (MAX_LEN,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (MAX_LEN,)
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
_CHUNK_RE = re.compile(r"chunk\d+")


def _build_input_text(row: pd.Series, sep: str) -> str:
    """Concatenate skill + all chunk columns with the tokenizer's separator token.

    Reads however many chunk1..chunkN columns the CSV has (set by
    config.RETRIEVE_TOP_K at build time), in numeric order — no hard-coded count.
    `sep` is the model's own separator (DeBERTa: '[SEP]').
    """
    chunk_cols = sorted(
        (c for c in row.index if _CHUNK_RE.fullmatch(str(c))),
        key=lambda c: int(str(c)[5:]),
    )
    chunks = " ".join(
        str(row[c]) for c in chunk_cols
        if pd.notna(row[c]) and str(row[c]).strip()
    )
    return f"{row['skill']} {sep} {chunks}"


def _split_frames(df: pd.DataFrame):
    """Stratified train/val/test split that preserves original row indices."""
    labels = df["label"]

    train_df, temp_df = train_test_split(
        df,
        test_size=config.VAL_RATIO + config.TEST_RATIO,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )

    relative_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=config.RANDOM_SEED,
        stratify=temp_df["label"],
    )
    return train_df, val_df, test_df


def _make_loader(df: pd.DataFrame, tokenizer, shuffle: bool) -> DataLoader:
    sep = tokenizer.sep_token
    texts = df.apply(lambda r: _build_input_text(r, sep), axis=1).tolist()
    labels = (df["label"] - 1).tolist()  # shift 1-5 → 0-4
    ds = ScoringDataset(texts, labels, tokenizer)
    return DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,      # parallel tokenization
        pin_memory=torch.cuda.is_available(),  # faster host->GPU copies
        persistent_workers=config.NUM_WORKERS > 0,
    )


def _subsample(df: pd.DataFrame, subset: float) -> pd.DataFrame:
    """Stratified down-sample for quick smoke runs.

    `subset` in (0, 1) is treated as a fraction; >= 1 as an absolute row count.
    Sampling is per-label so every class is still represented (a stratified
    split needs >= 2 rows per class). Original row indices are preserved so the
    retrieval-meta join in evaluate.py still works on subset runs.
    """
    if subset >= 1:
        frac = min(1.0, float(subset) / len(df))
    else:
        frac = float(subset)
    sampled = df.groupby("label", group_keys=False).sample(
        frac=frac, random_state=config.RANDOM_SEED
    )
    return sampled


def load_data(subset: float | None = None):
    """
    Read CSV → build text inputs → split into train / val / test.

    Parameters
    ----------
    subset : optional fraction (0-1) or row count (>=1) for a fast smoke run.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    train_loader, val_loader, test_loader, _ = load_data_with_frames(subset=subset)
    return train_loader, val_loader, test_loader


def load_data_with_frames(subset: float | None = None):
    """Same as load_data() but also returns the test DataFrame.

    The test frame's index gives each test row's original CSV position, which
    evaluate.py uses to look up the row's retrieval provenance. The test loader
    is built un-shuffled from that same frame, so prediction order matches
    test_df row order exactly.

    Returns
    -------
    train_loader, val_loader, test_loader, test_df
    """
    df = pd.read_csv(config.DATA_PATH)

    # Drop data-quality-flagged rows (build_dataset rules 1-4). Index is preserved
    # so the retrieval-meta join in evaluate.py still works for the kept rows.
    if config.EXCLUDE_FLAGGED and "exclude" in df.columns:
        n_before = len(df)
        df = df[df["exclude"] == 0]
        print(f"[exclude] dropped {n_before - len(df)} flagged rows; {len(df)} remain")

    if subset is not None:
        df = _subsample(df, subset)
        print(f"[subset] Using {len(df)} rows "
              f"(label dist: {df['label'].value_counts().sort_index().to_dict()})")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_df, val_df, test_df = _split_frames(df)

    train_loader = _make_loader(train_df, tokenizer, shuffle=True)
    val_loader   = _make_loader(val_df,   tokenizer, shuffle=False)
    test_loader  = _make_loader(test_df,  tokenizer, shuffle=False)

    return train_loader, val_loader, test_loader, test_df
