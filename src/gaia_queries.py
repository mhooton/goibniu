import numpy as np
from astroquery.gaia import Gaia
from coordinate_utils import get_current_julian_year, propagate_position
import logging
logging.getLogger("astroquery").setLevel(logging.ERROR)

def get_field_jmag(gaia_id, config, expansion_factor=1.0):
    """
    Query Gaia DR2 for the target star and retrieve J-band magnitudes
    for all stars in the surrounding field.

    Args:
        gaia_id: Gaia DR2 source_id of the target star
        config: Configuration dictionary
        expansion_factor: Factor to expand FOV (1.0 = config FOV, >1.0 = larger)

    Returns:
        Astropy Table containing source_id, ra, dec, J-band mag, and Teff
        for all stars in the field
    """
    # Set Gaia table to DR2
    Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"

    # Get target star coordinates
    job = Gaia.launch_job_async(f"SELECT ra, dec FROM gaiadr2.gaia_source WHERE source_id={gaia_id}")
    result = job.get_results()
    if len(result) == 0:
        raise ValueError(f"Gaia ID {gaia_id} not found")
    ra, dec = result['ra'][0], result['dec'][0]

    # Define search box dimensions (in degrees) from config
    width = config['field_of_view']['width_arcmin'] / 60 * expansion_factor
    height = config['field_of_view']['height_arcmin'] / 60 * expansion_factor

    # Calculate half-widths for the box search
    half_width = width / 2.0
    half_height = height / 2.0

    # Account for RA compression at different declinations
    ra_half_width = half_width / abs(np.cos(np.radians(dec)))

    # Query for all stars in field with J-band magnitudes from 2MASS
    # Using BETWEEN instead of CONTAINS for better performance
    adql = f"""
    SELECT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.ref_epoch, tm.j_m, g.teff_val
    FROM gaiadr2.gaia_source AS g
    JOIN gaiadr2.tmass_best_neighbour AS xmatch
      ON g.source_id = xmatch.source_id
    JOIN gaiadr1.tmass_original_valid AS tm
      ON xmatch.tmass_oid = tm.tmass_oid
    WHERE g.ra BETWEEN {ra - ra_half_width} AND {ra + ra_half_width}
      AND g.dec BETWEEN {dec - half_height} AND {dec + half_height}
    ORDER BY tm.j_m ASC
    """

    job2 = Gaia.launch_job_async(adql)
    results = job2.get_results()

    # Propagate all positions to current epoch
    current_epoch = get_current_julian_year()

    ra_corrected = []
    dec_corrected = []

    for row in results:
        ra_new, dec_new = propagate_position(
            row['ra'],
            row['dec'],
            row['pmra'],
            row['pmdec'],
            row['ref_epoch'],
            current_epoch
        )
        ra_corrected.append(ra_new)
        dec_corrected.append(dec_new)

    # Update positions in the table
    results['ra'] = ra_corrected
    results['dec'] = dec_corrected

    return results