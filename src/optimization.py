import os
import numpy as np
from astropy.io import fits
import pandas as pd
from joblib import Parallel, delayed
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

from bad_pixel_handling import aperture_contains_bad_pixels, distance_to_nearest_hazard
from comparison_star_selection import calculate_expansion_factor, select_comparison_stars, aperture_on_detector
from coordinate_utils import create_wcs, sky_to_pixel, pixel_to_sky
from gaia_queries import get_field_jmag
from precision_prediction import convert_j_to_zyj, prediction_from_DT
from utils import to_float, create_run_directory
from visualization import save_precision_map_png

def optimize_target_position(gaia_id, config, bad_pixel_map, save_precision_map=False):
    """
    Find optimal detector position for target star to maximize photometric precision.

    Args:
        gaia_id: Gaia DR2 source_id of target star
        config: Configuration dictionary
        bad_pixel_map: 2D boolean array where True = bad pixel (passed in, not loaded)
        save_precision_map: If True, save 2D precision map as FITS file

    Returns:
        dict containing:
            - 'optimal_x': Best X pixel position
            - 'optimal_y': Best Y pixel position
            - 'precision': Predicted precision at optimal position
            - 'n_comparison_stars': Number of usable comparison stars
            - 'target_ra': Target RA (degrees)
            - 'target_dec': Target Dec (degrees)
            - 'precision_map': 2D array of precision values (if save_precision_map=True)
    """
    logger.info("\n=== Starting Position Optimization ===")

    # Show BPM information
    bpm_path = config['detector']['bad_pixel_map_path']
    bpm_filename = os.path.basename(bpm_path)
    n_bad_pixels = np.sum(bad_pixel_map)
    logger.info(f"Bad pixel map: {bpm_filename}")
    logger.info(f"Using bad pixel map: {n_bad_pixels} bad pixels ({n_bad_pixels / bad_pixel_map.size * 100:.2f}%)")

    # Query expanded field
    logger.info("Querying Gaia for expanded field...")
    expansion = calculate_expansion_factor(config)
    logger.info(f"Expansion factor: {expansion:.3f}")
    jmag_data = get_field_jmag(gaia_id, config, expansion_factor=expansion)
    logger.info(f"Gaia query returned {len(jmag_data)} stars")

    # Get target coordinates
    target_row = jmag_data[jmag_data['source_id'] == int(gaia_id)]
    if len(target_row) == 0:
        raise ValueError(f"Target star {gaia_id} not found in query results")
    target_ra = float(target_row['ra'][0])
    target_dec = float(target_row['dec'][0])
    target_jmag = float(target_row['j_m'][0])
    target_teff = to_float(target_row['teff_val'][0])

    logger.info(f"Target: RA={target_ra:.6f}, Dec={target_dec:.6f}, J={target_jmag:.3f}, Teff={target_teff:.0f}K")

    # Get detector parameters
    det_width = config['detector']['width_pixels']
    det_height = config['detector']['height_pixels']
    aperture_radius = config['aperture']['radius_pixels']
    edge_padding = config['detector']['edge_padding_pixels']
    grid_spacing = config['optimization']['grid_spacing_pixels']

    logger.info(f"Detector: {det_width}×{det_height} px, aperture={aperture_radius} px, padding={edge_padding} px")
    logger.info(f"Grid spacing: {grid_spacing} px")

    # Filter comparison stars by magnitude (relative to target)
    fainter_limit = config['comparison_star_limits']['fainter_limit']
    brighter_limit = config['comparison_star_limits']['brighter_limit']

    logger.info(f"Magnitude limits: target J={target_jmag:.3f}, range=[{target_jmag + brighter_limit:.3f},"
                 f"{target_jmag+ fainter_limit:.3f}]")


    comp_stars = jmag_data[
        (jmag_data['source_id'] != int(gaia_id)) &
        (jmag_data['j_m'] < target_jmag + fainter_limit) &
        (jmag_data['j_m'] > target_jmag + brighter_limit)
        ]

    logger.info(f"Potential comparison stars in magnitude range: {len(comp_stars)}")

    if len(comp_stars) == 0:
        logger.info("No comparison stars found in magnitude range!")
        logger.info("Cannot create precision map without comparison stars.")
        raise ValueError("Cannot optimize without comparison stars")

    # Create grid of target positions
    x_positions = np.arange(aperture_radius + edge_padding,
                            det_width - aperture_radius - edge_padding,
                            grid_spacing)
    y_positions = np.arange(aperture_radius + edge_padding,
                            det_height - aperture_radius - edge_padding,
                            grid_spacing)

    total_positions = len(x_positions) * len(y_positions)
    logger.info(f"Testing {total_positions} grid positions (spacing={grid_spacing} pixels)...")

    # Store results
    results = []

    # Convert target J-mag to zYJ once
    target_zyj = to_float(convert_j_to_zyj(target_jmag, config))

    # Run parallel grid search
    logger.info(f"Running parallel grid search using all available CPU cores...")

            # Check if target aperture is clean
            if aperture_contains_bad_pixels(target_x, target_y, aperture_radius, bad_pixel_map):
                n_target_bad_pix += 1
                continue

            # Create WCS for this target position
            wcs_obj = create_wcs(target_ra, target_dec, target_x, target_y, config)

            # Transform all comparison stars to pixel coordinates
            comp_x, comp_y = sky_to_pixel(comp_stars['ra'], comp_stars['dec'], wcs_obj)

            # Select comparison stars for this position
            comp_selection = select_comparison_stars(
                target_x, target_y,
                comp_x, comp_y, comp_stars['j_m'],
                bad_pixel_map, config
            )

            # Skip if no valid comparison stars
            if comp_selection['n_valid'] == 0:
                n_no_valid_comps += 1
                continue

            # Predict precision using Decision Tree
            features = pd.DataFrame({
                'Comp stars': [comp_selection['n_valid']],
                'zYJ mag': [target_zyj],
                'Combined mag': [to_float(comp_selection['combined_mag'])],
                'Teff': [target_teff]
            })
            precision = to_float(prediction_from_DT(features))

    logger.info(f"\n=== Optimization Statistics ===")
    logger.info(f"Total positions tested: {n_tested}")
    logger.info(
        f"Target aperture off detector: {n_target_off_detector} ({n_target_off_detector / n_tested * 100:.1f}%)")
    logger.info(f"Target aperture contains bad pixels: {n_target_bad_pix} ({n_target_bad_pix / n_tested * 100:.1f}%)")
    logger.info(f"No valid comparison stars: {n_no_valid_comps} ({n_no_valid_comps / n_tested * 100:.1f}%)")
    logger.info(f"Successful positions: {len(results)} ({len(results) / n_tested * 100:.1f}%)")

    if len(results) == 0:
        logger.info("\nNo valid positions found!")
        logger.info("\nPossible issues:")
        logger.info(f"  - Detector too small ({det_width}×{det_height} px)")
        logger.info(f"  - Aperture + padding too large ({aperture_radius + edge_padding} px from edges)")
        logger.info(f"  - All comparison stars fall outside detector at all positions")
        logger.info(f"  - Bad pixel contamination too high")
        raise ValueError("No valid positions found - detector may be too contaminated with bad pixels")

    # Find best precision
    precisions = np.array([r['precision'] for r in results])
    best_precision = np.min(precisions)

    # Get all positions with best precision
    optimal_positions = [r for r in results if r['precision'] == best_precision]

    logger.info(f"\nBest precision: {best_precision:.6f}")
    logger.info(f"Number of positions with best precision: {len(optimal_positions)}")
    logger.info(f"Precision range: [{np.min(precisions):.6f}, {np.max(precisions):.6f}]")
    logger.info(f"Mean precision: {np.mean(precisions):.6f}")

    # Break ties by maximizing distance to nearest hazard (bad pixel or edge)
    if len(optimal_positions) > 1:
        best_distance = -np.inf
        best_result = None

        for result in optimal_positions:
            dist = distance_to_nearest_hazard(result['x'], result['y'],
                                              bad_pixel_map, det_width, det_height, edge_padding)
            if dist > best_distance:
                best_distance = dist
                best_result = result
    else:
        best_result = optimal_positions[0]
        best_distance = distance_to_nearest_hazard(best_result['x'], best_result['y'],
                                                   bad_pixel_map, det_width, det_height, edge_padding)

    logger.info(f"\nOptimal position: X={best_result['x']:.1f}, Y={best_result['y']:.1f}")
    logger.info(f"Distance to nearest hazard (bad pixel or edge): {best_distance:.2f} pixels")
    logger.info(f"Number of comparison stars: {best_result['n_comp']}")

    # Calculate additional positioning information
    # Center of detector in detector coordinates
    center_x = det_width / 2.0
    center_y = det_height / 2.0

    # Offset of target from center in detector coordinates
    offset_x = best_result['x'] - center_x
    offset_y = best_result['y'] - center_y

    # Convert detector center to RA/Dec using optimal WCS
    wcs_optimal = create_wcs(target_ra, target_dec, best_result['x'], best_result['y'], config)
    center_ra, center_dec = pixel_to_sky(center_x, center_y, wcs_optimal)

    # Calculate offset in RA/Dec (target is at the reference position of the WCS)
    # The target RA/Dec is the WCS reference point, so offset from center:
    offset_ra = target_ra - center_ra  # degrees
    offset_dec = target_dec - center_dec  # degrees

    # Convert to arcseconds for readability
    offset_ra_arcsec = offset_ra * 3600.0
    offset_dec_arcsec = offset_dec * 3600.0

    logger.info(f"\n--- Detector Center Information ---")
    logger.info(f"Detector center (optimized): RA={center_ra:.6f}°, Dec={center_dec:.6f}°")
    logger.info(f"Target offset from center: ΔX={offset_x:.1f} px, ΔY={offset_y:.1f} px")
    logger.info(f"Target offset from center: ΔRA={offset_ra_arcsec:.2f}\", ΔDec={offset_dec_arcsec:.2f}\"")

    # Create precision map if requested
    precision_map = None
    if save_precision_map:
        logger.info("\n=== Creating Precision Map ===")

        # Calculate coarse grid dimensions
        coarse_height = len(y_positions)
        coarse_width = len(x_positions)

        logger.info(f"Coarse grid dimensions: {coarse_width} × {coarse_height}")
        logger.info(f"Each pixel represents {grid_spacing} × {grid_spacing} detector pixels")

        # Initialize coarse map with NaN
        precision_map = np.full((coarse_height, coarse_width), np.nan)

        # Fill in computed values at coarse grid positions
        # Map from detector coordinates to coarse grid indices
        x_pos_to_idx = {int(x): i for i, x in enumerate(x_positions)}
        y_pos_to_idx = {int(y): i for i, y in enumerate(y_positions)}

        for result in results:
            coarse_x_idx = x_pos_to_idx[int(result['x'])]
            coarse_y_idx = y_pos_to_idx[int(result['y'])]
            precision_map[coarse_y_idx, coarse_x_idx] = result['precision']

        n_filled = np.sum(np.isfinite(precision_map))
        logger.info(f"Precision map contains {n_filled} valid positions")

        # Generate output filename
        run_dir = create_run_directory(gaia_id)
        ref_image_base = 'noref'
        if 'reference_image' in config:
            ref_image_base = os.path.splitext(os.path.basename(config['reference_image']))[0]
        badpix_base = os.path.splitext(os.path.basename(config['detector']['bad_pixel_map_path']))[0]
        map_output_path = run_dir / f'precision_map_{gaia_id}_{ref_image_base}_{badpix_base}.fits'

        logger.info(f"Saving precision map FITS to: {map_output_path}")

        # Save to FITS with coarse grid parameters
        hdu = fits.PrimaryHDU(precision_map)
        hdu.header['BUNIT'] = 'precision'
        hdu.header['GAIA_ID'] = str(gaia_id)
        hdu.header['TARG_RA'] = (target_ra, 'Target RA (deg)')
        hdu.header['TARG_DEC'] = (target_dec, 'Target Dec (deg)')
        hdu.header['TARG_J'] = (target_jmag, 'Target J magnitude')
        hdu.header['TARG_TEFF'] = (target_teff, 'Target Teff (K)')
        hdu.header['OPT_X'] = (best_result['x'], 'Optimal X position (detector coords)')
        hdu.header['OPT_Y'] = (best_result['y'], 'Optimal Y position (detector coords)')
        hdu.header['OPT_PREC'] = (best_precision, 'Optimal precision')

        if 'reference_image' in config:
            hdu.header['REFIMAGE'] = (str(config['reference_image']), 'Reference FITS image')
        hdu.header['BADPXMAP'] = (str(config['detector']['bad_pixel_map_path']), 'Bad pixel map file')

        # Coarse grid parameters
        hdu.header['APERTURE'] = (aperture_radius, 'Aperture radius (detector pixels)')
        hdu.header['EDGEPAD'] = (edge_padding, 'Edge padding (detector pixels)')
        hdu.header['GRIDSPAC'] = (grid_spacing, 'Grid spacing (detector pixels)')
        hdu.header['PIXSCALE'] = (config['detector']['pixel_scale_arcsec'] * grid_spacing,
                                  'Pixel scale (arcsec/pixel, coarse grid)')
        hdu.header['DET_NX'] = (det_width, 'Detector width (detector pixels)')
        hdu.header['DET_NY'] = (det_height, 'Detector height (detector pixels)')

        hdu.header['COMPFNT'] = (fainter_limit, 'Comp star fainter limit (mag)')
        hdu.header['COMPBRT'] = (brighter_limit, 'Comp star brighter limit (mag)')
        hdu.header['NCOMPALL'] = (len(comp_stars), 'Total potential comp stars')
        hdu.header['NVALID'] = (len(results), 'Number of valid positions')

        hdu.header['COMMENT'] = 'Predicted photometric precision at grid positions'
        hdu.header['COMMENT'] = f'Coarse grid: each pixel = {grid_spacing}x{grid_spacing} detector pixels'
        hdu.header['COMMENT'] = 'NaN values indicate invalid positions'

        hdu.writeto(map_output_path, overwrite=True)

        logger.info(f"Precision map FITS saved successfully")

        # Create PNG visualization
        save_precision_map_png(precision_map, config, gaia_id,
                               best_result['x'], best_result['y'], best_precision,
                               det_width, det_height)

    target_jmag_val = float(target_jmag)

    return {
        'optimal_x': best_result['x'],
        'optimal_y': best_result['y'],
        'precision': best_result['precision'],
        'n_comparison_stars': best_result['n_comp'],
        'combined_mag': best_result['combined_mag'],
        'target_ra': target_ra,
        'target_dec': target_dec,
        'distance_to_hazard': best_distance,
        'detector_center_ra': center_ra,
        'detector_center_dec': center_dec,
        'target_offset_x': offset_x,
        'target_offset_y': offset_y,
        'target_offset_ra_arcsec': offset_ra_arcsec,
        'target_offset_dec_arcsec': offset_dec_arcsec,
        'target_jmag': target_jmag_val,
        'target_zyj': target_zyj,
        'target_teff': target_teff,
        'precision_map': precision_map
    }