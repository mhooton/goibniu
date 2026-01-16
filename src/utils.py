from datetime import datetime
from paths import RUNS_DIR

def to_float(value):
    """
    Convert various astropy/numpy types to plain Python float.

    Args:
        value: Quantity, masked array, numpy scalar, or numeric value

    Returns:
        Plain Python float
    """
    # Handle Quantity objects
    if hasattr(value, 'value'):
        value = value.value
    # Handle arrays/masked arrays - extract scalar
    if hasattr(value, 'item'):
        value = value.item()
    # Handle remaining masked array cases
    if hasattr(value, 'data'):
        value = value.data
        if hasattr(value, 'item'):
            value = value.item()
    return float(value)

def create_run_directory(gaia_id, output_dir=None):
    """
    Create output directory for this run.

    Args:
        gaia_id: Gaia DR2 source_id
        output_dir: Optional pre-created output directory path

    Returns:
        Absolute path to run directory
    """

    # If output_dir provided (e.g., from batch mode), use it
    if output_dir is not None:
        return output_dir

    date_str = datetime.now().strftime("%Y%m%d")
    run_dir = RUNS_DIR / f"{gaia_id}_{date_str}"

    # Create directory (overwrite if exists)
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir
