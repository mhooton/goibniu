from datetime import datetime
import csv
import json
from paths import PROJECT_ROOT, RUNS_DIR

def read_target_list(filename):
    """
    Read Gaia DR2 IDs from target list file.

    Expects a space/comma-separated file with Gaia IDs in the 3rd column.
    Skips lines starting with # or that don't have at least 3 columns.

    Args:
        filename: Name of file in target_lists/ directory

    Returns:
        List of Gaia DR2 IDs as integers
    """
    target_list_path = PROJECT_ROOT / "target_lists" / filename

    if not target_list_path.exists():
        raise FileNotFoundError(f"Target list not found: {target_list_path}")

    gaia_ids = []

    with open(target_list_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Split by whitespace or comma
            parts = line.replace(',', ' ').split()

            if len(parts) >= 3:
                try:
                    gaia_id = int(parts[2])  # 3rd column (0-indexed: 2)
                    gaia_ids.append(gaia_id)
                except ValueError:
                    # Skip lines where 3rd column isn't a valid integer
                    continue

    print(f"Read {len(gaia_ids)} targets from {filename}")
    return gaia_ids


def create_batch_directory():
    """
    Create batch output directory.

    Returns:
        Absolute path to batch directory
    """

    date_str = datetime.now().strftime("%Y%m%d")
    batch_dir = RUNS_DIR / f"batch_{date_str}"

    # Create directory
    batch_dir.mkdir(parents=True, exist_ok=True)

    return batch_dir


def initialize_batch_csv(batch_dir):
    """
    Create and initialize batch summary CSV file.

    Args:
        batch_dir: Path to batch directory

    Returns:
        Path to CSV file
    """

    date_str = datetime.now().strftime("%Y%m%d")
    csv_path = batch_dir / f"batch_summary_{date_str}.csv"

    # Define CSV headers
    headers = [
        'gaia_id',
        'status',
        'timestamp',
        'target_ra',
        'target_dec',
        'optimal_x',
        'optimal_y',
        'detector_center_ra',
        'detector_center_dec',
        'target_offset_x',
        'target_offset_y',
        'target_offset_ra_arcsec',
        'target_offset_dec_arcsec',
        'precision',
        'n_comparison_stars',
        'combined_mag',
        'target_jmag',
        'target_zyj',
        'target_teff',
        'distance_to_hazard',
        'reference_image',
        'bad_pixel_map',
        'processing_time_seconds',
        'error_message'
    ]

    # Write header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    return csv_path


def append_to_batch_csv(csv_path, result):
    """
    Append a result row to the batch CSV.

    Args:
        csv_path: Path to CSV file
        result: Dictionary with result data
    """
    row = [
        result.get('gaia_id', ''),
        result.get('status', ''),
        result.get('timestamp', ''),
        result.get('target_ra', ''),
        result.get('target_dec', ''),
        result.get('optimal_x', ''),
        result.get('optimal_y', ''),
        result.get('detector_center_ra', ''),
        result.get('detector_center_dec', ''),
        result.get('target_offset_x', ''),
        result.get('target_offset_y', ''),
        result.get('target_offset_ra_arcsec', ''),
        result.get('target_offset_dec_arcsec', ''),
        result.get('precision', ''),
        result.get('n_comparison_stars', ''),
        result.get('combined_mag', ''),
        result.get('target_jmag', ''),
        result.get('target_zyj', ''),
        result.get('target_teff', ''),
        result.get('distance_to_hazard', ''),
        result.get('reference_image', ''),
        result.get('bad_pixel_map', ''),
        result.get('processing_time_seconds', ''),
        result.get('error_message', '')
    ]

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_batch_metadata(batch_dir, date_str, target_list_file, config):
    """
    Save batch metadata for resumption consistency.

    Args:
        batch_dir: Path to batch directory
        date_str: Batch date string (YYYYMMDD)
        target_list_file: Target list filename
        config: Configuration dictionary
    """

    metadata = {
        'batch_date': date_str,
        'created_timestamp': datetime.now().isoformat(),
        'target_list': target_list_file,
        'bad_pixel_map': str(config['detector']['bad_pixel_map_path']),
        'reference_image': str(config.get('reference_image', '')),
        'aperture_radius': config['aperture']['radius_pixels'],
        'grid_spacing': config['optimization']['grid_spacing_pixels'],
        'edge_padding': config['detector']['edge_padding_pixels'],
        'detector_width': config['detector']['width_pixels'],
        'detector_height': config['detector']['height_pixels']
    }

    metadata_path = batch_dir / 'batch_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Batch metadata saved to: {metadata_path}")


def load_and_validate_batch_metadata(batch_dir, config, force_config=False):
    """
    Load batch metadata and validate against current config.

    Args:
        batch_dir: Path to batch directory
        config: Current configuration dictionary
        force_config: If True, skip validation and use original config

    Returns:
        metadata dictionary

    Raises:
        ValueError if validation fails and force_config=False
    """

    metadata_path = batch_dir / 'batch_metadata.json'

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Batch metadata not found: {metadata_path}\n"
            f"Cannot resume batch - metadata file is missing."
        )

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\n=== Batch Resume Validation ===")
    print(f"Original batch date: {metadata['batch_date']}")
    print(f"Target list: {metadata['target_list']}")
    print(f"Bad pixel map: {metadata['bad_pixel_map']}")
    print(f"Reference image: {metadata['reference_image']}")

    # Validate critical parameters
    mismatches = []

    if config['aperture']['radius_pixels'] != metadata['aperture_radius']:
        mismatches.append(
            f"Aperture radius: {config['aperture']['radius_pixels']} vs {metadata['aperture_radius']}"
        )

    if config['optimization']['grid_spacing_pixels'] != metadata['grid_spacing']:
        mismatches.append(
            f"Grid spacing: {config['optimization']['grid_spacing_pixels']} vs {metadata['grid_spacing']}"
        )

    if config['detector']['edge_padding_pixels'] != metadata['edge_padding']:
        mismatches.append(
            f"Edge padding: {config['detector']['edge_padding_pixels']} vs {metadata['edge_padding']}"
        )

    if config['detector']['width_pixels'] != metadata['detector_width']:
        mismatches.append(
            f"Detector width: {config['detector']['width_pixels']} vs {metadata['detector_width']}"
        )

    if config['detector']['height_pixels'] != metadata['detector_height']:
        mismatches.append(
            f"Detector height: {config['detector']['height_pixels']} vs {metadata['detector_height']}"
        )

    if mismatches:
        error_msg = "\nERROR: Configuration mismatch detected:\n"
        for mismatch in mismatches:
            error_msg += f"  - {mismatch}\n"
        error_msg += "\nCannot resume batch with different parameters.\n"
        error_msg += "Use --force-config to override (not recommended)."

        if not force_config:
            raise ValueError(error_msg)
        else:
            print("\nWARNING: Configuration mismatch detected but proceeding due to --force-config:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")

    return metadata


def read_completed_targets(csv_path, skip_failed=False):
    """
    Read list of completed Gaia IDs from batch CSV.

    Args:
        csv_path: Path to batch CSV file
        skip_failed: If True, include FAILED targets in completed list (skip retrying)

    Returns:
        Set of Gaia IDs that should be skipped
    """


    if not csv_path.exists():
        return set()

    completed_ids = set()
    failed_ids = set()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gaia_id = int(row['gaia_id'])
            status = row['status']

            if status == 'SUCCESS':
                completed_ids.add(gaia_id)
            elif status == 'FAILED':
                failed_ids.add(gaia_id)

    if skip_failed:
        # Skip both successful and failed targets
        return completed_ids | failed_ids
    else:
        # Only skip successful targets (retry failed ones)
        return completed_ids
