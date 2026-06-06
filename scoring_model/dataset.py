"""
dataset.py – Data loading, preprocessing, and train/val/test splitting
for the CORAL ordinal-regression scoring model.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split

import config


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class ScoringDataset(Dataset):
    """
    Reads rows with columns:  skill, chunk1, chunk2, chunk3, label
    • Concatenates input as:  "{skill} </s> {chunk1} {chunk2} {chunk3}"
    • Tokenises with RoBERTa tokenizer (pad / truncate to MAX_LEN)
    • Shifts labels from 1-5 → 0-4 (zero-indexed for CORAL)
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer: RobertaTokenizer):
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
def _build_input_text(row: pd.Series) -> str:
    """Concatenate skill + chunks with RoBERTa separator token."""
    chunks = " ".join(
        str(row[c]) for c in ("chunk1", "chunk2", "chunk3") if pd.notna(row[c])
    )
    return f"{row['skill']} </s> {chunks}"


def load_data():
    """
    Read CSV → build text inputs → split into train / val / test.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    df = pd.read_csv(config.DATA_PATH)

    texts = df.apply(_build_input_text, axis=1).tolist()
    labels = (df["label"] - 1).tolist()  # shift 1-5 → 0-4

    tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME)

    # ── stratified splits ───────────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels,
        test_size=config.VAL_RATIO + config.TEST_RATIO,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )

    relative_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        random_state=config.RANDOM_SEED,
        stratify=y_temp,
    )

    train_ds = ScoringDataset(X_train, y_train, tokenizer)
    val_ds   = ScoringDataset(X_val,   y_val,   tokenizer)
    test_ds  = ScoringDataset(X_test,  y_test,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=config.BATCH_SIZE)

    return train_loader, val_loader, test_loader
