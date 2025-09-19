from pathlib import Path
from numpy import geomspace, array, linspace

SEED = 44
RADIUS = 3
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
OUTPUT_DIR = BASE_DIR / "data/output"
PROCESSED_DIR = BASE_DIR / "data/processed"
MODEL_NAMES = ["cnn", "gnn", "fcd", "fcfp", "cb"]
SEED = 42
DEFAULT_DATA_OFFSET = 1.5
DEFAULT_DATA_OFFSETS = array((0, 1.5, 10))
DEFAULT_THRESHOLD = 5.0
DEFAULT_THRESHOLDS = geomspace(0.1, 100, 15)
TOL = 1e-8
