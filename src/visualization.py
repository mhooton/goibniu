import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Patch
from matplotlib.colors import Normalize
import os
from astroquery.skyview import SkyView
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

from comparison_star_selection import calculate_expansion_factor
from coordinate_utils import pixel_to_sky, sky_to_pixel, create_wcs
from gaia_queries import get_field_jmag, get_target_properties
from utils import create_run_directory

def create_optimization_visualization(gaia_id, config, opt_result, output_path=None):
    """
    Create a PNG visualization showing the queried field with sky survey background,
    stars, and optimal detector position.

    Args:
        gaia_id: Gaia DR2 source_id of target star
        config: Configuration dictionary
        opt_result: Dictionary returned from optimize_target_position()
        output_path: Path to output PNG file (if None, auto-generate)
    """

    logger.info("\n=== Creating Optimization Visualization ===")

    # Auto-generate output filename if not provided
    if output_path is None:
        run_dir = create_run_directory(gaia_id, output_dir=output_path)
        ref_image_base = 'noref'
        if 'reference_image' in config:
            ref_image_base = os.path.splitext(os.path.basename(config['reference_image']))[0]
        badpix_base = os.path.splitext(os.path.basename(config['detector']['bad_pixel_map_path']))[0]
        output_path = run_dir / f'optimization_viz_{gaia_id}_{ref_image_base}_{badpix_base}.png'

    # Query expanded field for comparison stars
    expansion = calculate_expansion_factor(config)
    jmag_data = get_field_jmag(gaia_id, config, expansion_factor=expansion)

    # Get target info from opt_result — avoids redundant Gaia query
    target_ra = opt_result['target_ra']
    target_dec = opt_result['target_dec']
    optimal_x = opt_result['optimal_x']
    optimal_y = opt_result['optimal_y']
    target_jmag = opt_result['target_jmag']

    # Filter comparison stars by magnitude
    fainter_limit = config['comparison_star_limits']['fainter_limit']
    brighter_limit = config['comparison_star_limits']['brighter_limit']
    comp_stars = jmag_data[
        (jmag_data['source_id'] != int(gaia_id)) &
        (jmag_data['j_m'] < target_jmag + fainter_limit) &
        (jmag_data['j_m'] > target_jmag + brighter_limit)
        ]

    # Calculate field size for sky survey query
    all_ra = jmag_data['ra']
    all_dec = jmag_data['dec']
    ra_range = np.max(all_ra) - np.min(all_ra)
    dec_range = np.max(all_dec) - np.min(all_dec)

    # Add margin
    margin_factor = 1.1
    ra_width = ra_range * margin_factor
    dec_height = dec_range * margin_factor

    # Center position
    center_ra = np.mean(all_ra)
    center_dec = np.mean(all_dec)

    logger.info(f"Querying sky survey at RA={center_ra:.6f}, Dec={center_dec:.6f}")
    logger.info(f"Field size: {ra_width * 60:.2f}' × {dec_height * 60:.2f}'")

    # Query sky survey image
    try:
        center_coord = SkyCoord(center_ra, center_dec, unit='deg', frame='icrs')
        img_list = SkyView.get_images(
            position=center_coord,
            survey='DSS2 Red',
            width=ra_width * u.deg,
            height=dec_height * u.deg,
            pixels=[1000, 1000]
        )

        if len(img_list) == 0:
            raise ValueError("No image returned from SkyView")

        survey_hdu = img_list[0][0]
        survey_data = survey_hdu.data
        survey_wcs = WCS(survey_hdu.header)

        logger.info("Sky survey image retrieved successfully")

    except Exception as e:
        logger.info(f"Warning: Could not retrieve sky survey image: {e}")
        logger.info("Creating visualization without background image")
        survey_data = None
        survey_wcs = None

    # Create WCS for optimal detector position
    detector_wcs = create_wcs(target_ra, target_dec, optimal_x, optimal_y, config)

    # Get detector parameters
    det_width = config['detector']['width_pixels']
    det_height = config['detector']['height_pixels']
    aperture_radius = config['aperture']['radius_pixels']
    pixel_scale_arcsec = config['detector']['pixel_scale_arcsec']

    # Create figure with proportional dimensions based on RA/Dec range
    aspect_ratio = (ra_width * np.cos(np.radians(center_dec))) / dec_height
    if aspect_ratio >= 1:
        fig_width = 14
        fig_height = 14 / aspect_ratio
    else:
        fig_height = 14
        fig_width = 14 * aspect_ratio

    # Create figure with WCS projection
    if survey_wcs is not None:
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = plt.subplot(projection=survey_wcs)

        # Plot sky survey image
        ax.imshow(survey_data, origin='lower', cmap='gray',
                  vmin=np.percentile(survey_data, 5),
                  vmax=np.percentile(survey_data, 99.5))

        # Invert the Y-axis to match detector orientation (south-up)
        ax.invert_yaxis()

        # Configure RA/Dec display
        ax.coords[0].set_format_unit(u.deg)
        ax.coords[1].set_format_unit(u.deg)
        ax.coords[0].set_major_formatter('d.ddd')
        ax.coords[1].set_major_formatter('d.ddd')

        # Set equal aspect for RA/Dec (accounting for declination)
        # ax.set_aspect('equal')  # This should work with WCS projection

    else:
        # Fallback without survey image
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_facecolor('black')
        ax.set_aspect('equal')

    # Plot all queried stars
    ax.plot(all_ra, all_dec, 'o', color='cyan', markersize=3,
            alpha=0.6, label='All queried stars', transform=ax.get_transform('world'))

    # Plot comparison stars with circles and full Gaia IDs
    for ra, dec, source_id in zip(comp_stars['ra'], comp_stars['dec'], comp_stars['source_id']):
        # Convert aperture radius to sky coordinates (approximate)
        aperture_radius_deg = aperture_radius * pixel_scale_arcsec / 3600.0

        # Draw circle around comparison star
        circle = Circle((ra, dec), aperture_radius_deg, fill=False,
                        edgecolor='blue', linewidth=1.5, alpha=0.8,
                        transform=ax.get_transform('world'))
        ax.add_patch(circle)

        # Add star marker
        ax.plot(ra, dec, '*', color='blue', markersize=10, alpha=0.9,
                transform=ax.get_transform('world'))

        # Add full Gaia ID label
        label = str(source_id)
        ax.annotate(label, xy=(ra, dec), xycoords=ax.get_transform('world'),
                    xytext=(15, 0), textcoords='offset points',  # 15 points to the right
                    fontsize=6, color='blue', alpha=0.8,
                    verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.7, edgecolor='blue'))

    # Plot target star
    aperture_radius_deg = aperture_radius * pixel_scale_arcsec / 3600.0
    target_circle = Circle((target_ra, target_dec), aperture_radius_deg,
                           fill=False, edgecolor='red', linewidth=2.5, alpha=0.95,
                           transform=ax.get_transform('world'))
    ax.add_patch(target_circle)

    ax.plot(target_ra, target_dec, '*', color='red', markersize=15,
            alpha=0.95, label='Target star', transform=ax.get_transform('world'))

    # Add target label with full Gaia ID
    target_label = f"Target\n{gaia_id}"
    ax.annotate(target_label, xy=(target_ra, target_dec), xycoords=ax.get_transform('world'),
                xytext=(15, 0), textcoords='offset points',  # 15 points to the right
                fontsize=8, color='red', weight='bold',
                verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.8, edgecolor='red'))

    # Draw detector boundary at optimal position
    # Get detector center from optimization result
    detector_center_ra = opt_result['detector_center_ra']
    detector_center_dec = opt_result['detector_center_dec']

    # Draw detector boundary at optimal position
    # Get detector corners in detector coordinates (X, Y order)
    detector_corners_x = np.array([0, det_width, det_width, 0, 0])
    detector_corners_y = np.array([0, 0, det_height, det_height, 0])

    # Convert to RA/Dec using the detector WCS
    # This WCS has target at position (optimal_x, optimal_y)
    detector_corners_ra, detector_corners_dec = pixel_to_sky(
        detector_corners_x, detector_corners_y, detector_wcs
    )

    # Debug: print corner positions
    logger.info(f"Detector corners RA range: [{np.min(detector_corners_ra):.6f}, {np.max(detector_corners_ra):.6f}]")
    logger.info(f"Detector corners Dec range: [{np.min(detector_corners_dec):.6f}, {np.max(detector_corners_dec):.6f}]")
    logger.info(f"Target RA/Dec: ({target_ra:.6f}, {target_dec:.6f})")
    logger.info(f"Detector center RA/Dec: ({detector_center_ra:.6f}, {detector_center_dec:.6f})")

    # Plot detector FOV
    ax.plot(detector_corners_ra, detector_corners_dec,
            color='lime', linewidth=2.5, linestyle='--',
            alpha=0.9, label='Detector FOV (optimal)',
            transform=ax.get_transform('world'))

    # Mark detector center with X
    ax.plot(detector_center_ra, detector_center_dec, 'x',
            color='lime', markersize=18, markeredgewidth=3, alpha=0.9,
            transform=ax.get_transform('world'))

    # Mark detector center with X
    detector_center_ra = opt_result['detector_center_ra']
    detector_center_dec = opt_result['detector_center_dec']
    ax.plot(detector_center_ra, detector_center_dec, 'x',
            color='lime', markersize=18, markeredgewidth=3, alpha=0.9,
            transform=ax.get_transform('world'))

    # Labels and title
    if survey_wcs is not None:
        ax.coords[0].set_axislabel('RA (deg)', fontsize=12)
        ax.coords[1].set_axislabel('Dec (deg)', fontsize=12)
    else:
        ax.set_xlabel('RA (deg)', fontsize=12)
        ax.set_ylabel('Dec (deg)', fontsize=12)

    ax.set_title(f'Optimal Detector Positioning - Gaia DR2 {gaia_id}\n'
                 f'Precision: {opt_result["precision"]:.6f} | '
                 f'Comparison stars: {opt_result["n_comparison_stars"]}',
                 fontsize=14, weight='bold')

    # Grid
    if survey_wcs is not None:
        ax.coords.grid(color='white', alpha=0.3, linestyle=':', linewidth=0.5)
    else:
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='white')

    # Create secondary axes for Detector X/Y coordinates
    # We need to create transformation functions with bounds checking
    def ra_to_detx(ra_vals):
        """Transform RA to detector X coordinate"""
        if isinstance(ra_vals, (int, float)):
            ra_vals = np.array([ra_vals])
        # For each RA, use the mean Dec to get approximate X
        dec_vals = np.full_like(ra_vals, target_dec)
        x_vals, _ = sky_to_pixel(ra_vals, dec_vals, detector_wcs)
        # Replace any invalid values with reasonable bounds
        x_vals = np.where(np.isfinite(x_vals), x_vals,
                          np.where(x_vals > 0, det_width, 0))
        return x_vals

    def detx_to_ra(x_vals):
        """Transform detector X to RA"""
        if isinstance(x_vals, (int, float)):
            x_vals = np.array([x_vals])
        # For each X, use center Y to get RA
        y_vals = np.full_like(x_vals, optimal_y)
        ra_vals, _ = pixel_to_sky(x_vals, y_vals, detector_wcs)
        # Ensure finite values
        ra_vals = np.where(np.isfinite(ra_vals), ra_vals, target_ra)
        return ra_vals

    def dec_to_dety(dec_vals):
        """Transform Dec to detector Y coordinate"""
        if isinstance(dec_vals, (int, float)):
            dec_vals = np.array([dec_vals])
        # For each Dec, use the mean RA to get approximate Y
        ra_vals = np.full_like(dec_vals, target_ra)
        _, y_vals = sky_to_pixel(ra_vals, dec_vals, detector_wcs)
        # Replace any invalid values with reasonable bounds
        y_vals = np.where(np.isfinite(y_vals), y_vals,
                          np.where(y_vals > 0, det_height, 0))
        return y_vals

    def dety_to_dec(y_vals):
        """Transform detector Y to Dec"""
        if isinstance(y_vals, (int, float)):
            y_vals = np.array([y_vals])
        # For each Y, use center X to get Dec
        x_vals = np.full_like(y_vals, optimal_x)
        _, dec_vals = pixel_to_sky(x_vals, y_vals, detector_wcs)
        # Ensure finite values
        dec_vals = np.where(np.isfinite(dec_vals), dec_vals, target_dec)
        return dec_vals

    # # Create secondary X axis (top) for detector X
    # try:
    #     ax_top = ax.secondary_xaxis('top', functions=(ra_to_detx, detx_to_ra))
    #     ax_top.set_xlabel('Detector X (pixels)', fontsize=12)
    # except Exception as e:
    #     print(f"Warning: Could not create secondary X axis: {e}")
    #
    # # Create secondary Y axis (right) for detector Y
    # try:
    #     ax_right = ax.secondary_yaxis('right', functions=(dec_to_dety, dety_to_dec))
    #     ax_right.set_ylabel('Detector Y (pixels)', fontsize=12)
    # except Exception as e:
    #     print(f"Warning: Could not create secondary Y axis: {e}")

    # Legend
    ax.legend(loc='upper left', fontsize=10, fancybox=True,
              framealpha=0.8)

    # Add info text box
    info_text = (f"Aperture radius: {aperture_radius} px\n"
                 f"Detector: {det_width}×{det_height} px\n"
                 f"Distance to hazard: {opt_result['distance_to_hazard']:.1f} px\n"
                 f"Optimal position: ({optimal_x:.1f}, {optimal_y:.1f})\n"
                 f"Target offset: ({opt_result['target_offset_x']:.1f}, {opt_result['target_offset_y']:.1f}) px\n"
                 f"Detector center: ({opt_result['detector_center_ra']:.6f}°, {opt_result['detector_center_dec']:.6f}°)")
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            color='black')

    try:
        plt.tight_layout()
    except ValueError:
        logger.info("Warning: tight_layout failed, saving without layout adjustment")

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Visualization saved to: {output_path}")

def save_precision_map_png(precision_map, config, gaia_id, optimal_x_det, optimal_y_det,
                          best_precision, det_width, det_height):
    """
    Create a PNG visualization of the precision map.

    Args:
        precision_map: 2D array of precision values (coarse grid)
        config: Configuration dictionary
        gaia_id: Gaia DR2 source_id
        optimal_x_det: Optimal X position (detector coordinates)
        optimal_y_det: Optimal Y position (detector coordinates)
        best_precision: Best precision value
        det_width: Full detector width (detector pixels)
        det_height: Full detector height (detector pixels)
    """


    logger.info("\n=== Creating Precision Map PNG ===")

    # Get parameters
    coarse_height, coarse_width = precision_map.shape
    edge_padding = config['detector']['edge_padding_pixels']
    aperture_radius = config['aperture']['radius_pixels']
    grid_spacing = config['optimization']['grid_spacing_pixels']

    # Calculate vmin and vmax
    finite_mask = np.isfinite(precision_map)
    if not np.any(finite_mask):
        logger.info("ERROR: No finite precision values in map")
        return

    finite_values = precision_map[finite_mask]
    vmin = np.min(finite_values)
    vmax = np.max(finite_values)

    logger.info(f"Precision range: [{vmin:.6f}, {vmax:.6f}]")

    # Create figure with proportional dimensions
    # Base size on detector aspect ratio
    aspect_ratio = det_width / det_height
    if aspect_ratio >= 1:
        # Wider than tall
        fig_width = 12
        fig_height = 12 / aspect_ratio
    else:
        # Taller than wide
        fig_height = 12
        fig_width = 12 * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create RGB image manually to handle NaN coloring
    cmap = plt.colormaps.get_cmap('viridis')
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Initialize RGB array
    rgb_image = np.ones((coarse_height, coarse_width, 3))

    # Fill in valid precision values with colormap
    for i in range(coarse_height):
        for j in range(coarse_width):
            if np.isfinite(precision_map[i, j]):
                rgb_image[i, j, :3] = cmap(norm(precision_map[i, j]))[:3]
            else:
                # NaN pixels (bad pixels or no comparison stars) -> red
                rgb_image[i, j] = [1.0, 0.0, 0.0]

    # Display the image with extent in detector coordinates
    # The coarse grid starts at (first_x, first_y) and ends at (last_x, last_y)
    first_x = edge_padding + aperture_radius
    last_x = det_width - edge_padding - aperture_radius
    first_y = edge_padding + aperture_radius
    last_y = det_height - edge_padding - aperture_radius

    im = ax.imshow(rgb_image, origin='lower',
                   extent=[first_x, last_x, first_y, last_y],
                   aspect='equal', interpolation='nearest')

    # Draw edge padding box (safe zone boundary) - in detector coordinates
    padding_rect = Rectangle(
        (edge_padding, edge_padding),
        det_width - 2 * edge_padding,
        det_height - 2 * edge_padding,
        linewidth=2, edgecolor='black', facecolor='none',
        linestyle='--', label='Edge padding boundary'
    )
    ax.add_patch(padding_rect)

    # Mark optimal position with a cross - in detector coordinates
    ax.plot(optimal_x_det, optimal_y_det, 'wx', markersize=20, markeredgewidth=3,
            label=f'Optimal position ({optimal_x_det:.1f}, {optimal_y_det:.1f})')

    # Axis labels (detector coordinates)
    ax.set_xlabel('Detector X Position (pixels)', fontsize=12)
    ax.set_ylabel('Detector Y Position (pixels)', fontsize=12)
    ax.set_title(f'Precision Map - Gaia DR2 {gaia_id}\n'
                f'Grid spacing: {grid_spacing} px | Best precision: {best_precision:.6f}',
                fontsize=14, weight='bold')

    # Set axis limits to full detector dimensions (shows untested edge regions)
    ax.set_xlim(0, det_width)
    ax.set_ylim(0, det_height)

    # Add colorbar for valid precision values
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Precision', fontsize=12)

    # Add legend
    legend_elements = [
        ax.get_legend_handles_labels()[0][0],  # Edge padding box
        ax.get_legend_handles_labels()[0][1],  # Optimal position
        Patch(facecolor='red', edgecolor='black', label='Invalid (bad pixels/no comps)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='white')

    # Generate output filename
    run_dir = create_run_directory(gaia_id)
    ref_image_base = 'noref'
    if 'reference_image' in config:
        ref_image_base = os.path.splitext(os.path.basename(config['reference_image']))[0]
    badpix_base = os.path.splitext(os.path.basename(config['detector']['bad_pixel_map_path']))[0]
    png_output_path = run_dir / f'precision_map_{gaia_id}_{ref_image_base}_{badpix_base}.png'

    try:
        plt.tight_layout()
    except ValueError:
        logger.info("Warning: tight_layout failed, saving without layout adjustment")

    plt.savefig(png_output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Precision map PNG saved to: {png_output_path}")
