import numpy as np
import paramiko
from datetime import datetime, timedelta
import os
from astropy.io import fits
from paths import BPM_DIR
from credentials import SFTP_HOST, SFTP_USERNAME, SFTP_BASE_PATH, SFTP_TELESCOPE


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
    Load bad pixel map from FITS file.
    Downloads from server if configured, otherwise uses local path.

    Args:
        config: Configuration dictionary

    Returns:
        2D boolean numpy array where True = bad pixel, False = good pixel
    """

    # Check if we should download from server
    if config.get('download_BPM_from_server', False):
        bpm_path = download_latest_bpm(config)

        if bpm_path is None:
            # Fall back to config path
            print(f"Falling back to configured BPM path: {config['detector']['bad_pixel_map_path']}")
            bpm_path = config['detector']['bad_pixel_map_path']
        else:
            # Update config with the downloaded path
            config['detector']['bad_pixel_map_path'] = bpm_path
    else:
        bpm_path = config['detector']['bad_pixel_map_path']

    with fits.open(bpm_path) as hdul:
        # FITS convention: 1 = bad, 0 = good
        bad_pixel_data = hdul[0].data

    # Convert to boolean: True = bad pixel
    return bad_pixel_data.astype(bool)

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
