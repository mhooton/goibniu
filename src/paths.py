# src/paths.py
from pathlib import Path

# Get the absolute path to the spirit_precision directory
# Assumes paths.py is in spirit_precision/src/
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

# Define standard directories
CONFIG_DIR = PROJECT_ROOT / "configs"
BPM_DIR = PROJECT_ROOT / "BPMs"
REF_IMAGE_DIR = PROJECT_ROOT / "ref_images"
RUNS_DIR = PROJECT_ROOT / "runs"
MODEL_PATH = CONFIG_DIR / "precision_prediction_model.joblib"

# Ensure directories exist
BPM_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)