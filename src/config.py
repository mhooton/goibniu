import numpy as np
import json
from astropy.io import fits
from pathlib import Path
from paths import CONFIG_DIR, BPM_DIR, REF_IMAGE_DIR

def load_config(config_path=None):
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to JSON config file (if None, uses default)

    Returns:
        Dictionary containing configuration parameters
    """
    if config_path is None:
        config_path = CONFIG_DIR / "config.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Convert relative paths in config to absolute paths
    if 'reference_image' in config:
        ref_path = Path(config['reference_image'])
        if not ref_path.is_absolute():
            config['reference_image'] = str(REF_IMAGE_DIR / ref_path)

    if 'bad_pixel_map_path' in config.get('detector', {}):
        bpm_path = Path(config['detector']['bad_pixel_map_path'])
        if not bpm_path.is_absolute():
            config['detector']['bad_pixel_map_path'] = str(BPM_DIR / bpm_path)

    # If reference_image is provided, extract detector/WCS parameters from it
    if 'reference_image' in config:
        print(f"Loading detector parameters from: {config['reference_image']}")
        fits_params = load_detector_from_fits(config['reference_image'])

        # Merge FITS-derived parameters into config
        if 'detector' not in config:
            config['detector'] = {}

        config['detector'].update(fits_params['detector'])
        config['wcs'] = fits_params['wcs']
        config['field_of_view'] = fits_params['field_of_view']

    return config

def load_detector_from_fits(fits_path):
    """
    Extract detector and WCS parameters from a plate-solved FITS image.

    Args:
        fits_path: Path to plate-solved FITS file

    Returns:
        Dictionary containing detector and WCS parameters
    """

    with fits.open(fits_path) as hdul:
        header = hdul[0].header

    # Extract detector dimensions
    width_pixels = header['NAXIS1']
    height_pixels = header['NAXIS2']

    # Extract WCS parameters
    pc1_1 = header['PC1_1']
    pc1_2 = header['PC1_2']
    pc2_1 = header['PC2_1']
    pc2_2 = header['PC2_2']
    cdelt1 = header.get('CDELT1', 1.0)
    cdelt2 = header.get('CDELT2', 1.0)

    # Calculate pixel scale in arcsec/pixel
    # Use the PC matrix magnitude and CDELT
    pixel_scale_deg = np.sqrt(pc1_1 ** 2 + pc1_2 ** 2) * abs(cdelt1)
    pixel_scale_arcsec = pixel_scale_deg * 3600.0

    # Calculate field of view in arcminutes
    fov_width_arcsec = width_pixels * pixel_scale_arcsec
    fov_height_arcsec = height_pixels * pixel_scale_arcsec
    fov_width_arcmin = fov_width_arcsec / 60.0
    fov_height_arcmin = fov_height_arcsec / 60.0

    print(f"\n=== Detector Parameters from FITS ===")
    print(f"Dimensions: {width_pixels} × {height_pixels} pixels")
    print(f"Pixel scale: {pixel_scale_arcsec:.4f} arcsec/pixel")
    print(f"Field of view: {fov_width_arcmin:.2f}' × {fov_height_arcmin:.2f}'")

    return {
        'detector': {
            'width_pixels': width_pixels,
            'height_pixels': height_pixels,
            'pixel_scale_arcsec': pixel_scale_arcsec
        },
        'wcs': {
            'pc1_1': pc1_1,
            'pc1_2': pc1_2,
            'pc2_1': pc2_1,
            'pc2_2': pc2_2
        },
        'field_of_view': {
            'width_arcmin': fov_width_arcmin,
            'height_arcmin': fov_height_arcmin
        }
    }