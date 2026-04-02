"""
Central configuration for Distracted Driver Detection system.
All thresholds, class mappings, and model/training settings live here.
"""

from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models" / "weights"

# ─── Driver Behaviour Classes (State Farm dataset) ────────────────────────────
CLASS_NAMES: dict[str, str] = {
    "c0": "safe_driving",
    "c1": "texting_right",
    "c2": "phone_right",
    "c3": "texting_left",
    "c4": "phone_left",
    "c5": "radio_operating",
    "c6": "drinking",
    "c7": "reaching_behind",
    "c8": "hair_makeup",
    "c9": "talking_to_passenger",
}

CLASS_INDEX: dict[str, int] = {k: i for i, k in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS: dict[int, str] = {i: k for i, k in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# ─── Risk Level Definitions ───────────────────────────────────────────────────
# Raw base risk score (0-100) per class — used by risk_calculator.py
CLASS_BASE_RISK: dict[str, float] = {
    "c0": 0.0,    # safe driving
    "c1": 95.0,   # texting right — eyes off road
    "c2": 80.0,   # phone call right — partial cognitive load
    "c3": 95.0,   # texting left — eyes off road
    "c4": 80.0,   # phone call left
    "c5": 30.0,   # radio — brief manual distraction
    "c6": 65.0,   # drinking — hand + eyes distraction
    "c7": 70.0,   # reaching behind — severe posture distraction
    "c8": 55.0,   # hair/makeup — moderate eyes off road
    "c9": 20.0,   # talking to passenger — cognitive only
}

# Sustained distraction multiplier schedule (seconds → multiplier)
SUSTAINED_RISK_SCHEDULE: list[tuple[float, float]] = [
    (0.0, 1.0),
    (2.0, 1.2),
    (5.0, 1.5),
    (10.0, 2.0),
    (20.0, 2.5),
]

# Alert thresholds (composite risk score 0-100)
RISK_THRESHOLDS: dict[str, float] = {
    "LOW":      20.0,
    "MEDIUM":   45.0,
    "HIGH":     70.0,
    "CRITICAL": 85.0,
}

# ─── Model / Training ─────────────────────────────────────────────────────────
MODEL_BACKBONE = "mobilenet_v2"       # lightweight — suitable for edge
IMAGE_SIZE: tuple[int, int] = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
PRETRAINED = True                     # ImageNet pretrained backbone

# Data split ratios
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

# Normalisation (ImageNet statistics)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ─── Inference / Pipeline ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.50          # minimum confidence to emit a detection
ONNX_MODEL_NAME = "driver_classifier.onnx"
PT_MODEL_NAME   = "driver_classifier.pt"

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_IMAGE_SIZE_MB = 5
