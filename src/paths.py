# src/paths.py
import json
from pathlib import Path

# Get the absolute path to the spirit_precision directory
# Assumes paths.py is in spirit_precision/src/
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

# Define standard directories
CONFIG_DIR = PROJECT_ROOT / "configs"
BPM_DIR = PROJECT_ROOT / "BPMs"
REF_IMAGE_DIR = PROJECT_ROOT / "ref_images"
MODEL_PATH = CONFIG_DIR / "precision_prediction_model.joblib"

# Resolve RUNS_DIR from config.json if present, otherwise default to PROJECT_ROOT / "runs"
_config_path = CONFIG_DIR / "config.json"
_runs_dir_override = None
if _config_path.exists():
    with open(_config_path, 'r') as _f:
        _cfg = json.load(_f)
    _runs_dir_override = _cfg.get('runs_dir')

if _runs_dir_override:
    RUNS_DIR = Path(_runs_dir_override)
else:
    RUNS_DIR = PROJECT_ROOT / "runs"