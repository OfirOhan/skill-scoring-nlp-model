# ──────────────────────────────────────────────────────────────
# Scoring-model hyper-parameters  (single source of truth)
# ──────────────────────────────────────────────────────────────

# ── Transformer backbone ────────────────────────────────────
MODEL_NAME = "roberta-base"
MAX_LEN = 512

# ── Training ────────────────────────────────────────────────
BATCH_SIZE = 16
EPOCHS = 15
MLP_LR = 1e-3
ROBERTA_LR = 2e-5
DROPOUT = 0.3
FREEZE_ROBERTA = True

# ── Data ────────────────────────────────────────────────────
DATA_PATH = "data.csv"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ── CORAL ───────────────────────────────────────────────────
NUM_CLASSES = 5          # ordinal labels 1-5  →  0-4 (zero-indexed)
NUM_CORAL_OUTPUTS = NUM_CLASSES - 1   # = 4 cumulative logits

# ── Checkpoint ──────────────────────────────────────────────
CHECKPOINT_PATH = "best_model.pt"
