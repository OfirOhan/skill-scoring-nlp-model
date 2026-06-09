# ──────────────────────────────────────────────────────────────
# Scoring-model hyper-parameters  (single source of truth)
# ──────────────────────────────────────────────────────────────

from pathlib import Path

# ── Paths (absolute, cwd-independent) ───────────────────────
_THIS_DIR = Path(__file__).resolve().parent          # scoring_model/
_PROJECT_DIR = _THIS_DIR.parent                       # repo root
_DATA_DIR = _PROJECT_DIR / "data"

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
# Built by scoring_model/build_dataset.py (RAG → training rows).
DATA_PATH = str(_DATA_DIR / "training_data.csv")
# Per-row retrieval provenance + ground-truth evidence, written alongside
# the CSV by the same builder. Consumed by evaluate.py for retrieval metrics.
RETRIEVAL_META_PATH = str(_DATA_DIR / "retrieval_meta.jsonl")

# Source artifacts produced by the generation pipeline (run_generation.py).
PERSONAS_PATH = str(_DATA_DIR / "personas.json")
DOCUMENTS_DB_PATH = str(_DATA_DIR / "documents_db.json")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ── Retrieval (dataset construction) ────────────────────────
RETRIEVE_TOP_K = 3       # chunks fed to the scorer (chunk1, chunk2, chunk3)
EVIDENCE_THRESHOLD = 2   # min skill_evidence intensity for a doc to count
                         # as a *relevant* retrieval target (see build_dataset)

# ── CORAL ───────────────────────────────────────────────────
NUM_CLASSES = 5          # ordinal labels 1-5  →  0-4 (zero-indexed)
NUM_CORAL_OUTPUTS = NUM_CLASSES - 1   # = 4 cumulative logits

# ── Checkpoint ──────────────────────────────────────────────
CHECKPOINT_PATH = str(_THIS_DIR / "best_model.pt")
