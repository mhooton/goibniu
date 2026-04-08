import numpy as np
from coordinate_utils import sky_to_pixel, create_wcs
from bad_pixel_handling import aperture_contains_bad_pixels
from precision_prediction import convert_j_to_zyj, combined_mag

def calculate_expansion_factor(config):
    """
    Calculate FOV expansion factor needed to cover all detector positions.

    When optimizing, the target can be placed anywhere on the detector (within
    edge constraints). The worst case is target at one edge - we need to query
    far enough to include comparison stars that could appear at the opposite edge
    of the detector.

    Args:
        config: Configuration dictionary

    Returns:
        Expansion factor (>= 1.0)
    """
    width_pix = config['detector']['width_pixels']
    height_pix = config['detector']['height_pixels']
    pixel_scale = config['detector']['pixel_scale_arcsec']
    edge_padding = config['detector']['edge_padding_pixels']
    aperture_radius = config['aperture']['radius_pixels']

    # Maximum distance from target position to detector edge
    # (target can be placed at edge + padding + aperture constraints)
    max_distance_x = width_pix - (edge_padding + aperture_radius)
    max_distance_y = height_pix - (edge_padding + aperture_radius)

    # Query needs to extend this far in all directions from target
    # So total query size is 2× this distance
    query_width_arcsec = 2 * max_distance_x * pixel_scale
    query_height_arcsec = 2 * max_distance_y * pixel_scale

    # Calculate expansion relative to configured FOV
    width_expansion = query_width_arcsec / (config['field_of_view']['width_arcmin'] * 60)
    height_expansion = query_height_arcsec / (config['field_of_view']['height_arcmin'] * 60)

    # Use the larger expansion factor and add small safety margin
    expansion = max(width_expansion, height_expansion) * 1.01

    return max(1.0, expansion)

def aperture_on_detector(x, y, radius, detector_width, detector_height, padding):
    """
    Check if aperture is fully on detector with padding.

    Args:
        x: Aperture center X coordinate (pixel)
        y: Aperture center Y coordinate (pixel)
        radius: Aperture radius (pixels)
        detector_width: Detector width (pixels)
        detector_height: Detector height (pixels)
        padding: Edge padding (pixels)

    Returns:
        True if aperture is fully on detector (with padding), False otherwise
    """
    min_coord = radius + padding
    max_x = detector_width - radius - padding
    max_y = detector_height - radius - padding

    return (x >= min_coord and x <= max_x and
            y >= min_coord and y <= max_y)


def select_comparison_stars(target_x_det, target_y_det,
                            comp_x, comp_y, comp_jmags,
                            bad_pixel_map, config, n_comparison=4):
    """
    Select optimal comparison stars for differential photometry at a given detector position.

    This function filters comparison stars based on:
    1. Detector positioning (aperture must be fully on detector)
    2. Bad pixel contamination (aperture must be clean)
    3. Minimum separation from target (optional, via config)

    Args:
        target_x_det: Target X position in detector coordinates (pixels)
        target_y_det: Target Y position in detector coordinates (pixels)
        comp_x: Array of comparison star X positions in detector coordinates (pixels)
        comp_y: Array of comparison star Y positions in detector coordinates (pixels)
        comp_jmags: Array of comparison star J-band magnitudes
        bad_pixel_map: 2D boolean array where True = bad pixel
        config: Configuration dictionary containing:
                - detector: width_pixels, height_pixels, edge_padding_pixels
                - aperture: radius_pixels
                - comparison_star_limits: (optional) min_separation_pixels
        n_comparison: Maximum number of comparison stars to select (if None, return all valid)

    Returns:
        dict containing:
            - 'selected_stars': List of indices into comp_stars for selected comparison stars
            - 'combined_mag': Combined J-band magnitude of selected comparison stars (in zYJ)
            - 'n_valid': Number of valid comparison stars found
            - 'pixel_positions': List of (x, y) tuples for selected stars in detector coords
            - 'j_magnitudes': List of J-band magnitudes for selected stars
    """
    # Extract config parameters
    det_width = config['detector']['width_pixels']
    det_height = config['detector']['height_pixels']
    aperture_radius = config['aperture']['radius_pixels']
    edge_padding = config['detector']['edge_padding_pixels']

    # Optional minimum separation
    min_separation = config.get('comparison_star_limits', {}).get('min_separation_pixels', 0)

    # Filter comparison stars: on detector, clean apertures, and minimum separation
    valid_comps = []
    valid_comp_positions = []
    valid_comp_mags = []

    for j, (cx, cy, jmag) in enumerate(zip(comp_x, comp_y, comp_jmags)):
        # Check minimum separation from target (if specified)
        if min_separation > 0:
            separation = np.sqrt((cx - target_x_det) ** 2 + (cy - target_y_det) ** 2)
            if separation < min_separation:
                continue

        # Check if on detector
        if not aperture_on_detector(cx, cy, aperture_radius, det_width, det_height, edge_padding):
            continue

        # Check if aperture is clean
        if aperture_contains_bad_pixels(cx, cy, aperture_radius, bad_pixel_map):
            continue

        valid_comps.append(j)
        valid_comp_positions.append((cx, cy))
        valid_comp_mags.append(jmag)

    # If n_comparison is specified and we have more valid stars, select the brightest
    if n_comparison is not None and len(valid_comps) > n_comparison:
        # Sort by magnitude (brightest first)
        sorted_indices = np.argsort(valid_comp_mags)[:n_comparison]
        valid_comps = [valid_comps[i] for i in sorted_indices]
        valid_comp_positions = [valid_comp_positions[i] for i in sorted_indices]
        valid_comp_mags = [valid_comp_mags[i] for i in sorted_indices]

    # Calculate combined magnitude if we have valid comparison stars
    combined_mag_val = None
    if len(valid_comps) > 0:
        valid_comp_mags = np.array(valid_comp_mags)
        # Convert J magnitudes to zYJ
        comp_zyj = convert_j_to_zyj(valid_comp_mags, config)
        # Calculate combined magnitude
        combined_mag_val = combined_mag(comp_zyj)

    return {
        'selected_stars': valid_comps,
        'combined_mag': combined_mag_val,
        'n_valid': len(valid_comps),
        'pixel_positions': valid_comp_positions,
        'j_magnitudes': valid_comp_mags
    }
