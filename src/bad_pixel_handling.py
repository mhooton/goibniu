import numpy as np
import paramiko
from datetime import datetime, timedelta
import os
from astropy.io import fits
from paths import BPM_DIR
import re
from pathlib import Path
try:
    from credentials import SFTP_HOST, SFTP_USERNAME, SFTP_BASE_PATH, SFTP_TELESCOPE
    CREDENTIALS_AVAILABLE = True
except ImportError:
    CREDENTIALS_AVAILABLE = False
    SFTP_HOST = None
    SFTP_USERNAME = None
    SFTP_BASE_PATH = None
    SFTP_TELESCOPE = None

import logging

# Get logger for this module
logger = logging.getLogger(__name__)


def download_latest_bpm(config, cutoff_date="20210101"):
    """
    Download the latest bad pixel map from the SFTP server.

    Searches backwards from yesterday through dates, trying each version
    for each date until a BPM is found or cutoff date is reached.

    Args:
        config: Configuration dictionary (modified in-place to update BPM path)
        cutoff_date: Stop searching before this date (format: YYYYMMDD)
                     Raises error if no BPMs found after this date

    Returns:
        Path to the downloaded BPM file
    """
    print("\n=== Searching for latest Bad Pixel Map on server ===")

    # Parse cutoff date
    cutoff = datetime.strptime(cutoff_date, "%Y%m%d")

    # Start with yesterday
    current_date = datetime.now() - timedelta(days=1)

    versions = ["v2", "v3"]

    print(f"Connecting to {SFTP_HOST}...")

    # Create SSH client and use system SSH keys
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(SFTP_HOST, username=SFTP_USERNAME)
        sftp = ssh.open_sftp()
        print("Connected successfully")

        # Search backwards through dates
        while current_date >= cutoff:
            date_str = current_date.strftime("%Y%m%d")

            # Try each version for this date
            for version in versions:
                remote_path = f"{SFTP_BASE_PATH}/{version}/{SFTP_TELESCOPE}/output/{date_str}/reduction/1_BadPixelMap.fits"

                try:
                    # Check if file exists
                    sftp.stat(remote_path)

                    # File exists - download it
                    print(f"Found BPM: {version}/{date_str}")

                    local_filename = f"1_BadPixelMap_{date_str}.fits"
                    local_path = BPM_DIR / local_filename

                    print(f"Downloading to {local_path}...")
                    sftp.get(remote_path, str(local_path))
                    print(f"Successfully downloaded BPM from {version}/{date_str}")

                    # Update config to point to downloaded file
                    config['detector']['bad_pixel_map_path'] = str(local_path)

                    return local_path

                except IOError:
                    # File doesn't exist, try next version/date
                    continue

            # Move to previous day
            current_date -= timedelta(days=1)

        # If we get here, no BPM was found
        raise FileNotFoundError(
            f"No BPM files found after cutoff date {cutoff_date}. "
            f"Searched from yesterday back to {cutoff_date}."
        )

    finally:
        sftp.close()
        ssh.close()


def load_bad_pixel_map(config):
    """
    Load bad pixel map with priority hierarchy:
    1. Command-line override (handled before calling this)
    2. Config download_BPM_from_server
       - If True: download (requires credentials)
       - If False: use config['detector']['bad_pixel_map_path'] if exists
    3. Most recent dated BPM in BPMs directory
    4. Any BPM in BPMs directory
    5. Error if none found
    """
    bpm_path = None

    # Priority 1 & 2: Download or use config path
    if config.get('download_BPM_from_server', False):
        # Attempt download
        if CREDENTIALS_AVAILABLE and None not in [SFTP_HOST, SFTP_USERNAME, SFTP_BASE_PATH, SFTP_TELESCOPE]:
            try:
                bpm_path = download_latest_bpm(config)
                if bpm_path:
                    logger.info(f"Downloaded BPM: {bpm_path}")
            except Exception as e:
                logger.warning(f"BPM download failed: {e}")
        else:
            logger.warning("Credentials unavailable for BPM download")
    else:
        # Check config for explicit path
        if 'bad_pixel_map_path' in config.get('detector', {}):
            config_path = Path(config['detector']['bad_pixel_map_path'])
            if config_path.exists():
                bpm_path = config_path
                logger.info(f"Using BPM from config: {bpm_path}")

    # Priority 3 & 4: Find local BPM if needed
    if bpm_path is None:
        try:
            bpm_path = find_most_recent_bpm()
            logger.info(f"Using most recent local BPM: {bpm_path}")
        except FileNotFoundError as e:
            # Priority 5: Error
            raise FileNotFoundError(
                "No bad pixel map found. Please either:\n"
                "1. Enable BPM download with credentials, or\n"
                "2. Specify 'bad_pixel_map_path' in config, or\n"
                "3. Place a BPM file in the BPMs/ directory"
            )

    # Update config with final path
    config['detector']['bad_pixel_map_path'] = str(bpm_path)

    # Load the BPM
    with fits.open(bpm_path) as hdul:
        bad_pixel_data = hdul[0].data

    return bad_pixel_data.astype(bool)


def find_most_recent_bpm():
    """
    Find the most recent BPM file in the BPMs directory.

    Priority:
    1. Most recent dated file matching *BadPixelMap*YYYYMMDD.fits
    2. Any .fits file in directory
    3. Raise error if none found

    Returns:
        Path to BPM file
    """
    from paths import BPM_DIR
    import re
    from datetime import datetime

    # Find all FITS files
    all_fits_files = list(BPM_DIR.glob("*.fits"))

    if not all_fits_files:
        raise FileNotFoundError(f"No FITS files found in {BPM_DIR}")

    # Try to find dated BPM files (Priority 1)
    dated_files = []
    for filepath in all_fits_files:
        # Look for YYYYMMDD pattern in filename
        match = re.search(r'(\d{8})\.fits$', filepath.name)
        if match and 'BadPixelMap' in filepath.name:
            date_str = match.group(1)
            try:
                date = datetime.strptime(date_str, "%Y%m%d")
                dated_files.append((date, filepath))
            except ValueError:
                continue

    if dated_files:
        # Return most recent dated BPM
        most_recent = max(dated_files, key=lambda x: x[0])
        logger.info(f"Found dated BPM from {most_recent[0].strftime('%Y-%m-%d')}")
        return most_recent[1]

    # Priority 2: No dated files, return first FITS file found
    logger.warning(f"No dated BPM files found, using: {all_fits_files[0].name}")
    return all_fits_files[0]

def aperture_contains_bad_pixels(x, y, radius, bad_pixel_map):
    """
    Check if a circular aperture contains any bad pixels.

    Args:
        x: Aperture center X coordinate (pixel)
        y: Aperture center Y coordinate (pixel)
        radius: Aperture radius (pixels)
        bad_pixel_map: 2D boolean array where True = bad pixel

    Returns:
        True if aperture contains any bad pixels, False otherwise
    """
    # Get bounding box around aperture
    x_min = max(0, int(np.floor(x - radius)))
    x_max = min(bad_pixel_map.shape[1], int(np.ceil(x + radius)) + 1)
    y_min = max(0, int(np.floor(y - radius)))
    y_max = min(bad_pixel_map.shape[0], int(np.ceil(y + radius)) + 1)

    # Create coordinate grids for the bounding box
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

    # Calculate distances from aperture center
    distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    # Check if any bad pixels are within the aperture
    mask = distances <= radius
    if np.any(mask):
        return np.any(bad_pixel_map[y_min:y_max, x_min:x_max][mask])
    return False

def distance_to_nearest_hazard(x, y, bad_pixel_map, det_width, det_height, edge_padding):
    """
    Calculate distance from position to nearest hazard (bad pixel or detector edge).

    This is used as a tiebreaker among positions with equal precision - we prefer
    positions that are farther from both bad pixels and edges, maximizing the
    "safe zone" for small drifts.

    Args:
        x: Position X coordinate (pixel)
        y: Position Y coordinate (pixel)
        bad_pixel_map: 2D boolean array where True = bad pixel
        det_width: Detector width (pixels)
        det_height: Detector height (pixels)
        edge_padding: Edge padding (pixels)

    Returns:
        Distance to nearest hazard (pixels)
    """
    # Calculate distance to edges (with padding)
    dist_to_left = x - edge_padding
    dist_to_right = (det_width - edge_padding) - x
    dist_to_bottom = y - edge_padding
    dist_to_top = (det_height - edge_padding) - y

    dist_to_edge = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)

    # Find all bad pixel locations
    bad_y, bad_x = np.where(bad_pixel_map)

    if len(bad_y) == 0:
        # No bad pixels on detector - only edge distance matters
        return dist_to_edge

    # Calculate distances from (x, y) to all bad pixels
    distances_to_bad = np.sqrt((bad_x - x) ** 2 + (bad_y - y) ** 2)
    dist_to_bad_pixel = np.min(distances_to_bad)

    # Return the minimum of the two distances (nearest hazard)
    return min(dist_to_edge, dist_to_bad_pixel)
