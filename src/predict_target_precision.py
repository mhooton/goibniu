import numpy as np
import json
from datetime import datetime
import pandas as pd
import warnings
from pathlib import Path
import argparse
import time
warnings.filterwarnings('ignore', message='Trying to unpickle estimator')

import logging
# Setup logging
logger = logging.getLogger(__name__)

# Load modules
from config import load_config
from bad_pixel_handling import load_bad_pixel_map
from batch_processing import (read_target_list, save_batch_metadata, initialize_batch_csv, append_to_batch_csv,
                              load_and_validate_batch_metadata, read_completed_targets)
from gaia_queries import get_field_jmag
from optimization import optimize_target_position
from paths import RUNS_DIR
from precision_prediction import convert_j_to_zyj, combined_mag, effective_mg, prediction_from_fit, prediction_from_DT
from utils import create_run_directory, to_float
from visualization import create_optimization_visualization

def save_optimization_results(result, gaia_id, output_path=None):
    """
    Save optimization results to JSON file in run directory.

    Args:
        result: Dictionary returned from optimize_target_position()
        gaia_id: Gaia DR2 source_id
        output_path: Path to output JSON file (if None, auto-generate in run dir)
    """

    if output_path is None:
        run_dir = create_run_directory(gaia_id)
        output_path = run_dir / f"optimization_{gaia_id}.json"

    # Convert numpy types to Python native types for JSON serialization
    serializable_result = {
        'gaia_id': int(gaia_id),
        'status': result.get('status', 'SUCCESS'),
        'timestamp': result.get('timestamp', ''),
        'optimal_x': float(result['optimal_x']),
        'optimal_y': float(result['optimal_y']),
        'precision': float(result['precision']),
        'n_comparison_stars': int(result['n_comparison_stars']),
        'combined_mag': float(result['combined_mag']),
        'target_ra': float(result['target_ra']),
        'target_dec': float(result['target_dec']),
        'distance_to_hazard': float(result['distance_to_hazard']),
        'detector_center_ra': float(result['detector_center_ra']),
        'detector_center_dec': float(result['detector_center_dec']),
        'target_offset_x': float(result['target_offset_x']),
        'target_offset_y': float(result['target_offset_y']),
        'target_offset_ra_arcsec': float(result['target_offset_ra_arcsec']),
        'target_offset_dec_arcsec': float(result['target_offset_dec_arcsec']),
        'target_jmag': float(result.get('target_jmag', 0)),
        'target_zyj': float(result.get('target_zyj', 0)),
        'target_teff': float(result.get('target_teff', 0)),
        'reference_image': result.get('reference_image', ''),
        'bad_pixel_map': result.get('bad_pixel_map', ''),
        'processing_time_seconds': float(result.get('processing_time_seconds', 0)),
        'error_message': result.get('error_message', '')
    }

    with open(output_path, 'w') as f:
        json.dump(serializable_result, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def predict(gaia_id, config, bad_pixel_map, optimize=False, save_results=False,
            create_viz=False, save_precision_map=False):
    """
    Main prediction function for a given target star.

    Args:
        gaia_id: Gaia DR2 source_id of target star
        config: Configuration dictionary
        bad_pixel_map: 2D boolean array where True = bad pixel (passed in, not loaded)
        optimize: If True, optimize target position on detector. If False, assume centered.
        save_results: If True, save optimization results to JSON file
        create_viz: If True, create PNG visualization of optimization
        save_precision_map: If True, save precision map FITS file (only when optimize=True)
    """

    if optimize:
        # Run optimization to find best position (optionally creating precision map)
        start_time = time.time()
        opt_result = optimize_target_position(gaia_id, config, bad_pixel_map,
                                              save_precision_map=save_precision_map)
        processing_time = time.time() - start_time

        # Add metadata to result
        opt_result['gaia_id'] = gaia_id
        opt_result['status'] = 'SUCCESS'
        opt_result['timestamp'] = datetime.now().isoformat()
        opt_result['processing_time_seconds'] = processing_time
        opt_result['reference_image'] = config.get('reference_image', '')
        opt_result['bad_pixel_map'] = config['detector'].get('bad_pixel_map_path', '')
        opt_result['error_message'] = ''

        logger.info("\n=== Optimization Results ===")
        logger.info(f"Optimal detector position: X={opt_result['optimal_x']:.1f}, Y={opt_result['optimal_y']:.1f}")
        logger.info(f"Predicted precision: {opt_result['precision']:.6f}")
        logger.info(f"Usable comparison stars: {opt_result['n_comparison_stars']}")
        logger.info(f"Distance to nearest hazard: {opt_result['distance_to_hazard']:.2f} pixels")
        logger.info(f"Processing time: {processing_time:.1f} seconds")

        if save_results:
            save_optimization_results(opt_result, gaia_id)

        if create_viz:
            create_optimization_visualization(gaia_id, config, opt_result)

        return opt_result

    jmag_data = get_field_jmag(gaia_id, config)

    # Extract target star row
    target_row = jmag_data[jmag_data['source_id'] == int(gaia_id)]
    target_index = np.where(target_row)[0][0]

    # Filter comparison stars by magnitude range (from config)
    fainter_limit = config['comparison_star_limits']['fainter_limit']
    brighter_limit = config['comparison_star_limits']['brighter_limit']
    jmag_data = jmag_data[jmag_data['j_m'] < (target_row['j_m'] + fainter_limit)]
    jmag_data = jmag_data[jmag_data['j_m'] > (target_row['j_m'] + brighter_limit)]

    # Extract comparison star magnitudes (excluding target)
    comp_star_mag = jmag_data[jmag_data['source_id'] != int(gaia_id)]['j_m']

    # Convert all J-band magnitudes to zYJ
    zyj_mags_all = convert_j_to_zyj(jmag_data['j_m'], config)

    # Calculate combined magnitude (including target)
    combined_mags = combined_mag(zyj_mags_all)
    effective_mag = effective_mg(zyj_mags_all, target_index)

    # Predict precision using quadratic fit
    predicted_precision = 10 ** prediction_from_fit(combined_mags, config)
    eff_predicted_precision = 10 ** prediction_from_fit(effective_mag, config)

    # Calculate combined magnitude without target
    combined_mags_wo_target = combined_mag(convert_j_to_zyj(comp_star_mag, config))

    # Extract target properties and convert to floats
    n_comp = len(comp_star_mag)
    target_zyj = to_float(convert_j_to_zyj(target_row['j_m'], config))
    target_teff = to_float(target_row['teff_val'])
    combined_mags = to_float(combined_mags)
    effective_mag = to_float(effective_mag)
    combined_mags_wo_target = to_float(combined_mags_wo_target)
    predicted_precision = to_float(predicted_precision)

    # Print target information
    logger.info(f"\n--- Target Star Information ---")
    logger.info(f"Gaia DR2 ID: {gaia_id}")
    logger.info(f"zYJ magnitude: {target_zyj:.3f}")
    logger.info(f"Effective temperature: {target_teff:.0f} K")

    # Print field information
    logger.info(f"\n--- Field Information ---")
    logger.info(f"Number of comparison stars: {n_comp}")
    logger.info(f"Combined magnitude (with target): {combined_mags:.3f}")
    logger.info(f"Effective magnitude: {effective_mag:.3f}")
    logger.info(f"Combined magnitude (without target): {combined_mags_wo_target:.3f}")

    # Print quadratic fit prediction
    logger.info(f"\n--- Precision Predictions ---")
    logger.info(f"Quadratic fit combined prediction: {predicted_precision:.6f}")
    logger.info(f"Quadratic fit effective prediction: {eff_predicted_precision:.6f}")

    # Prepare features for Decision Tree prediction
    features = pd.DataFrame({
        'Comp stars': [n_comp],
        'zYJ mag': [target_zyj],
        'Combined mag': [combined_mags_wo_target],
        'Teff': [target_teff]
    })

    # Predict using Decision Tree model
    predicted_precision_DT = to_float(prediction_from_DT(features))
    logger.info(f"Decision Tree prediction: {predicted_precision_DT:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict photometric precision for SPECULOOS targets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single target
  python predict_target_precision.py --target 123456789 --optimize --save --viz

  # Batch processing
  python predict_target_precision.py --batch targets.txt --optimize --save --viz --map

  # Resume batch from specific date
  python predict_target_precision.py --batch targets.txt --resume 20250120 --optimize

  # Resume and skip failed targets
  python predict_target_precision.py --batch targets.txt --resume 20250120 --skip-failed --optimize
        """
    )

    # Target specification (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--target', type=int, help='Single Gaia DR2 source ID')
    target_group.add_argument('--batch', type=str, help='Batch target list filename (in target_lists/)')

    # Operation modes
    parser.add_argument('--optimize', action='store_true', help='Run position optimization')
    parser.add_argument('--centered', action='store_true', help='Run centered prediction')
    parser.add_argument('--save', action='store_true', help='Save results to JSON')
    parser.add_argument('--viz', action='store_true', help='Create visualization')
    parser.add_argument('--map', action='store_true', help='Create precision map')

    # Output verbosity
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=None,
                        help='Output verbosity: 0=quiet (one-line summaries only), '
                             '1=normal (detailed output), 2=verbose (includes parallel progress). '
                             'Defaults to 1 for single target, 0 for batch.')

    # Batch resume options
    parser.add_argument('--resume', type=str, metavar='YYYYMMDD',
                        help='Resume batch from specified date (format: YYYYMMDD)')
    parser.add_argument('--skip-failed', action='store_true',
                        help='Skip retrying failed targets when resuming (default: retry failed)')
    parser.add_argument('--force-config', action='store_true',
                        help='Override configuration validation errors when resuming')

    args = parser.parse_args()

    # Set verbosity default based on mode
    if args.verbosity is None:
        if args.batch:
            args.verbosity = 0  # Default to quiet for batch
        else:
            args.verbosity = 1  # Default to normal for single target

    # Configure logging based on verbosity
    if args.verbosity == 0:
        log_level = logging.WARNING
    elif args.verbosity == 1:
        log_level = logging.INFO
    else:  # verbosity == 2
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format='%(message)s'  # Clean format without timestamps/levels for user-facing output
    )

    # Load configuration once
    config = load_config()
    print("\n=== Loading Bad Pixel Map ===")
    bad_pixel_map = load_bad_pixel_map(config)
    n_bad_pixels = np.sum(bad_pixel_map)
    print(f"Bad pixel map loaded: {n_bad_pixels} bad pixels ({n_bad_pixels / bad_pixel_map.size * 100:.2f}%)")
    print(f"Using: {config['detector']['bad_pixel_map_path']}")

    # Single target mode (unchanged)
    if args.target:
        gaia_id = args.target

        logger.info("\n" + "=" * 60)
        logger.info(f"PROCESSING TARGET: {gaia_id}")
        logger.info("=" * 60)

        # Run centered prediction if requested
        if args.centered:
            logger.info("\n" + "=" * 60)
            logger.info("RUNNING CENTERED PREDICTION")
            logger.info("=" * 60)
            predict(gaia_id, config, bad_pixel_map, optimize=False, save_results=False,
                    create_viz=False)

        # Run optimization if requested
        if args.optimize:
            logger.info("\n" + "=" * 60)
            logger.info("RUNNING OPTIMIZATION")
            logger.info("=" * 60)
            predict(gaia_id, config, bad_pixel_map, optimize=True, save_results=args.save,
                    create_viz=args.viz, save_precision_map=args.map)

        # If no mode specified, default to centered prediction
        if not (args.centered or args.optimize):
            print("\n" + "=" * 60)
            print("RUNNING CENTERED PREDICTION (default)")
            print("=" * 60)
            predict(gaia_id, config, bad_pixel_map, optimize=False, save_results=False, create_viz=False)

    # Batch mode
    elif args.batch:
        from datetime import datetime

        # Determine batch date
        if args.resume:
            # Validate date format
            date_str = args.resume
            try:
                datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                print(f"Error: Invalid date format '{date_str}'. Use YYYYMMDD (e.g., 20250120)")
                return

            print("\n" + "=" * 60)
            print(f"RESUMING BATCH: {args.batch} from {date_str}")
            print("=" * 60)
        else:
            date_str = datetime.now().strftime("%Y%m%d")
            print("\n" + "=" * 60)
            print(f"BATCH PROCESSING: {args.batch}")
            print("=" * 60)

        # Read target list
        try:
            gaia_ids = read_target_list(args.batch)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        if len(gaia_ids) == 0:
            print("Error: No valid Gaia IDs found in target list")
            return

        # Create/access batch directory
        batch_dir = RUNS_DIR / f"batch_{date_str}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        csv_path = batch_dir / f"batch_summary_{date_str}.csv"

        # Handle resume vs new batch
        if args.resume:
            # Load and validate metadata
            try:
                metadata = load_and_validate_batch_metadata(batch_dir, config, args.force_config)
            except (FileNotFoundError, ValueError) as e:
                print(f"Error: {e}")
                return

            # Override config to use original BPM
            original_bpm_path = metadata['bad_pixel_map']
            config['detector']['bad_pixel_map_path'] = original_bpm_path
            config['download_BPM_from_server'] = False
            print(f"\nReloading original bad pixel map: {original_bpm_path}")
            bad_pixel_map = load_bad_pixel_map(config)
            n_bad_pixels = np.sum(bad_pixel_map)
            print(f"Bad pixel map reloaded: {n_bad_pixels} bad pixels ({n_bad_pixels / bad_pixel_map.size * 100:.2f}%)")

            # Read completed targets
            completed_ids = read_completed_targets(csv_path, skip_failed=args.skip_failed)

            n_completed = len(completed_ids)
            n_total = len(gaia_ids)
            n_remaining = n_total - n_completed

            print(f"\n=== Resume Summary ===")
            print(f"Total targets in list: {n_total}")
            print(f"Already completed: {n_completed}")
            print(f"Remaining to process: {n_remaining}")

            if args.skip_failed:
                print("Mode: Skipping failed targets")
            else:
                print("Mode: Retrying failed targets")

            # Filter target list
            gaia_ids = [gid for gid in gaia_ids if gid not in completed_ids]

            if len(gaia_ids) == 0:
                print("\nAll targets already completed. Nothing to do.")
                return

        else:
            # New batch - save metadata
            save_batch_metadata(batch_dir, date_str, args.batch, config)

            # Initialize new CSV
            csv_path = initialize_batch_csv(batch_dir)

        print(f"\nBatch directory: {batch_dir}")
        print(f"Results CSV: {csv_path}")
        print(f"Processing {len(gaia_ids)} targets...\n")

        # Process each target
        for i, gaia_id in enumerate(gaia_ids, 1):
            logger.info("\n" + "=" * 60)
            logger.info(f"TARGET {i}/{len(gaia_ids)}: {gaia_id}")
            logger.info("=" * 60)

            # try:
            # Run optimization
            result = predict(gaia_id, config, bad_pixel_map, optimize=True,
                             save_results=args.save, create_viz=args.viz,
                             save_precision_map=args.map)

            # Append to batch CSV
            append_to_batch_csv(csv_path, result)

                if args.verbosity == 0:
                    print(f"[{i}/{len(gaia_ids)}] {gaia_id} → SUCCESS (prec={result['precision']:.6f}, "
                          f"{result['n_comparison_stars']} comps, {result['processing_time_seconds']:.1f}s)")
                else:
                    logger.info(f"✓ Target {gaia_id} completed successfully")

            except Exception as e:
                # Handle failure
                if args.verbosity == 0:
                    print(f"[{i}/{len(gaia_ids)}] {gaia_id} → FAILED ({str(e)[:50]})")
                else:
                    logger.error(f"✗ Target {gaia_id} FAILED: {e}")

                # Write failure to CSV
                failed_result = {
                    'gaia_id': gaia_id,
                    'status': 'FAILED',
                    'timestamp': datetime.now().isoformat(),
                    'reference_image': config.get('reference_image', ''),
                    'bad_pixel_map': config['detector'].get('bad_pixel_map_path', ''),
                    'error_message': str(e)
                }
                append_to_batch_csv(csv_path, failed_result)

        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
