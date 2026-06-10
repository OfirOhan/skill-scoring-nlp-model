# ──────────────────────────────────────────────────────────────
# Scoring-model hyper-parameters  (single source of truth)
# ──────────────────────────────────────────────────────────────

from pathlib import Path

# ── Paths (absolute, cwd-independent) ───────────────────────
_THIS_DIR = Path(__file__).resolve().parent          # scoring_model/
_PROJECT_DIR = _THIS_DIR.parent                       # repo root
_DATA_DIR = _PROJECT_DIR / "data"

# ── Transformer backbone ────────────────────────────────────
MODEL_NAME = "microsoft/deberta-v3-base"
# Sized to fit all RETRIEVE_TOP_K(=8) chunks: ~1000-char chunks ≈ ~250 tokens
# each, so 8 chunks + skill ≈ ~2000 tokens. DeBERTa-v3 uses relative positions
# (no absolute-position cap), so it runs past its 512 pretraining length.
MAX_LEN = 2048

# ── Experiment selection (THE switch) ───────────────────────
# The 3 variants to compare. Each run saves to its own runs/<EXPERIMENT>/
# folder so attempts stay organized and directly comparable.
# unfreeze_last_n: how many of DeBERTa's 12 top transformer layers to fine-tune
# (0 = head-only; embeddings + lower layers always stay frozen). Full fine-tune
# is intentionally not an option — it overfits this small dataset.
EXPERIMENTS = {
    "coral_frozen":      {"head_type": "coral",      "unfreeze_last_n": 0,
                          "desc": "DeBERTa + CORAL head, head-only (current)"},
    "classifier_frozen": {"head_type": "classifier", "unfreeze_last_n": 0,
                          "desc": "DeBERTa + softmax classifier head, head-only"},
    "coral_top3":        {"head_type": "coral",      "unfreeze_last_n": 3,
                          "desc": "DeBERTa + CORAL head, fine-tune top 3 layers"},
}
EXPERIMENT = "coral_frozen"   # <-- change this (or pass --experiment) to switch

# Derived from EXPERIMENT (re-derived by apply_experiment at runtime).
HEAD_TYPE = None
UNFREEZE_LAST_N = None


def apply_experiment(name: str) -> None:
    """Select an experiment, updating the derived HEAD_TYPE / UNFREEZE_LAST_N."""
    global EXPERIMENT, HEAD_TYPE, UNFREEZE_LAST_N
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{name}'. Options: {list(EXPERIMENTS)}")
    EXPERIMENT = name
    HEAD_TYPE = EXPERIMENTS[name]["head_type"]
    UNFREEZE_LAST_N = EXPERIMENTS[name]["unfreeze_last_n"]


# ── Training ────────────────────────────────────────────────
# 8 fits the coral_top3 fine-tune at MAX_LEN=2048 on a 24 GB card (DeBERTa's
# disentangled attention at seq 2048 is memory-heavy). Frozen variants could go
# higher, but a single global value keeps the comparison clean.
BATCH_SIZE = 8
EPOCHS = 25
EARLY_STOP_PATIENCE = 10   # stop if val MAE hasn't improved for this many epochs
HEAD_LR = 1e-3       # MLP head learning rate
BACKBONE_LR = 2e-5   # transformer LR (used only when fine-tuning, i.e. unfrozen)
DROPOUT = 0.3
GRAD_CLIP = 1.0      # max grad norm — prevents fine-tuning blow-ups (NaN loss)

# ── GPU efficiency ──────────────────────────────────────────
NUM_WORKERS = 4      # DataLoader workers — parallelise the (slow) DeBERTa tokenizer
USE_AMP = True       # bf16 autocast on CUDA for forward passes (no-op on CPU)

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
# Number of reranked chunks retrieved, saved (chunk1..chunkN) and fed to the
# scorer. Single source of truth — build_dataset and dataset.py both follow it.
RETRIEVE_TOP_K = 8
EVIDENCE_THRESHOLD = 2   # min skill_evidence intensity for a doc to count
                         # as a *relevant* retrieval target (see build_dataset)

# ── Row exclusion (data-quality filters applied in build_dataset) ──
# Rows are tagged (not deleted) with an exclude_reason in retrieval_meta.jsonl;
# load_data drops excluded rows when EXCLUDE_FLAGGED is True, so thresholds can
# be toggled without rebuilding.
EXCLUDE_FLAGGED = True
SKILL_MISMATCH_MAX = 3   # rule 1: drop (persona,skill) if max |delta| >= this
SKILL_MISMATCH_AVG = 2   # rule 1: ...or if avg |delta| >= this
MIN_DOC_WORDS = 30       # rule 4: ingestion skips shorter / failed documents

_REPORTS_DIR = _DATA_DIR / "reports"
SHOWCASE_REPORT_PATH = str(_REPORTS_DIR / "skill_showcase_report.json")
ALLOCATION_REPORT_PATH = str(_REPORTS_DIR / "allocation_rationality_report.json")

# ── Classes ─────────────────────────────────────────────────
NUM_CLASSES = 5          # ordinal labels 1-5  →  0-4 (zero-indexed)
NUM_CORAL_OUTPUTS = NUM_CLASSES - 1   # = 4 cumulative logits (coral head only)

# ── Run artifacts (per experiment) ──────────────────────────
_RUNS_DIR = _THIS_DIR / "runs"

def run_dir() -> Path:
    """runs/<EXPERIMENT>/ — created on demand; holds the checkpoint + metrics."""
    d = _RUNS_DIR / EXPERIMENT
    d.mkdir(parents=True, exist_ok=True)
    return d

def checkpoint_path() -> str:
    return str(run_dir() / "best_model.pt")

def metrics_path() -> str:
    return str(run_dir() / "metrics.json")


# Initialise derived globals for the default experiment.
apply_experiment(EXPERIMENT)
