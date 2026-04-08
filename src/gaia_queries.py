import os
import sqlite3
import time
import numpy as np
import requests.exceptions
from astropy.table import Table
from coordinate_utils import get_current_julian_year, propagate_position
import logging
logging.getLogger("astroquery").setLevel(logging.ERROR)

def launch_job_async_with_retry(adql, max_retries=9, base_wait=15):
    """
    Wrapper around Gaia.launch_job_async with exponential backoff retry logic.
    Retries on transient server errors (500 and 404 job-not-found).

    Args:
        adql: ADQL query string
        max_retries: Maximum number of retry attempts
        base_wait: Base wait time in seconds (doubles each retry)

    Returns:
        Astropy Table containing query results
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            job = Gaia.launch_job_async(adql)
            return job.get_results()
        except requests.exceptions.HTTPError as e:
            error_str = str(e)
            print(str(e))
            if "500" in error_str or "service unavailable" in error_str.lower() or ("404" in error_str and "not found" in error_str.lower()):
                last_exception = e
                if attempt < max_retries:
                    wait_time = base_wait * (2 ** attempt)
                    logging.warning(
                        f"Gaia archive returned transient error (attempt {attempt + 1}/{max_retries + 1}): "
                        f"{error_str.strip()}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
            else:
                raise
    raise last_exception


def estimate_j_from_g_teff(g_mag, teff):
    """
    Estimate J-band magnitude from Gaia G magnitude and effective temperature
    using an empirical linear colour relation.

    Approximate relation valid for 2500K < Teff < 6500K, derived from
    Gaia-2MASS cross-matched samples of late-type stars. Accuracy is
    roughly +/-0.2 mag; treat results as approximate.

    Args:
        g_mag: Gaia G-band magnitude
        teff: Effective temperature in Kelvin

    Returns:
        Estimated J-band magnitude
    """
    # Empirical linear fit: (G - J) = 4.2 - 0.00058 * Teff
    # Gives G-J ~ 2.75 at 2500K (late M), ~ 0.72 at 6000K (G dwarf)
    g_minus_j = 4.2 - 0.00058 * teff
    return g_mag - g_minus_j


def _get_target_properties_archive(gaia_id):
    """
    Retrieve properties of the target star by Gaia DR2 source_id.

    Attempts to get J magnitude from 2MASS cross-match. If the target has
    no 2MASS match, falls back to estimating J from G magnitude and Teff.
    Fails hard if neither path is available.

    Args:
        gaia_id: Gaia DR2 source_id of the target star

    Returns:
        dict with keys: ra, dec, pmra, pmdec, ref_epoch, j_m, teff_val, j_estimated
            j_estimated: True if J was estimated from G+Teff, False if from 2MASS

    Raises:
        ValueError: If target not found, or J magnitude cannot be determined
    """
    from astroquery.gaia import Gaia

    # First query: attempt to get target with 2MASS J magnitude
    adql_tmass = f"""
    SELECT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.ref_epoch,
           g.phot_g_mean_mag, g.teff_val, tm.j_m
    FROM gaiadr2.gaia_source AS g
    LEFT JOIN gaiadr2.tmass_best_neighbour AS xmatch
      ON g.source_id = xmatch.source_id
    LEFT JOIN gaiadr1.tmass_original_valid AS tm
      ON xmatch.tmass_oid = tm.tmass_oid
    WHERE g.source_id = {gaia_id}
    """
    result = launch_job_async_with_retry(adql_tmass)

    if len(result) == 0:
        raise ValueError(f"Target star {gaia_id} not found in Gaia DR2")

    row = result[0]
    ra = float(row['ra'])
    dec = float(row['dec'])
    pmra = float(row['pmra']) if row['pmra'] is not None else 0.0
    pmdec = float(row['pmdec']) if row['pmdec'] is not None else 0.0
    ref_epoch = float(row['ref_epoch']) if row['ref_epoch'] is not None else 2015.5
    teff = row['teff_val']
    g_mag = row['phot_g_mean_mag']
    j_m = row['j_m']
    j_estimated = False

    # Check if we have a direct 2MASS J magnitude
    if j_m is not None and np.isfinite(float(j_m)):
        j_m = float(j_m)
        logging.info(f"Target {gaia_id}: using 2MASS J={j_m:.3f}")
    else:
        # Fall back to G+Teff estimation
        if teff is None or not np.isfinite(float(teff)):
            raise ValueError(
                f"Target star {gaia_id} has no 2MASS J magnitude and no Teff in Gaia DR2. "
                f"Cannot determine J magnitude."
            )
        if g_mag is None or not np.isfinite(float(g_mag)):
            raise ValueError(
                f"Target star {gaia_id} has no 2MASS J magnitude and no G magnitude in Gaia DR2. "
                f"Cannot determine J magnitude."
            )
        teff = float(teff)
        g_mag = float(g_mag)
        j_m = estimate_j_from_g_teff(g_mag, teff)
        j_estimated = True
        logging.warning(
            f"Target {gaia_id}: no 2MASS match found. "
            f"Estimated J={j_m:.3f} from G={g_mag:.3f}, Teff={teff:.0f}K. "
            f"Precision prediction will be approximate."
        )

    return {
        'ra': ra,
        'dec': dec,
        'pmra': pmra,
        'pmdec': pmdec,
        'ref_epoch': ref_epoch,
        'j_m': j_m,
        'teff_val': float(teff) if teff is not None else None,
        'j_estimated': j_estimated
    }


def _get_field_jmag_archive(gaia_id, config, expansion_factor=1.0):
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
    from astroquery.gaia import Gaia

    # Set Gaia table to DR2
    Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"

    # Get target coordinates from Gaia to define the search box
    target = get_target_properties(gaia_id)
    ra, dec = target['ra'], target['dec']

    # Define search box dimensions (in degrees) from config
    width = config['field_of_view']['width_arcmin'] / 60 * expansion_factor
    height = config['field_of_view']['height_arcmin'] / 60 * expansion_factor

    # Calculate half-widths for the box search
    half_width = width / 2.0
    half_height = height / 2.0

    # Account for RA compression at different declinations
    ra_half_width = half_width / abs(np.cos(np.radians(dec)))

    # Query for all comparison stars in field with confirmed 2MASS J magnitudes.
    # Target is excluded — its properties are handled separately by get_target_properties.
    # Inner joins intentional here: comparison stars without confirmed J mags are excluded.
    adql = f"""
        SELECT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.ref_epoch, tm.j_m, g.teff_val
        FROM gaiadr2.gaia_source AS g
        JOIN gaiadr2.tmass_best_neighbour AS xmatch
          ON g.source_id = xmatch.source_id
        JOIN gaiadr1.tmass_original_valid AS tm
          ON xmatch.tmass_oid = tm.tmass_oid
        WHERE g.ra BETWEEN {ra - ra_half_width} AND {ra + ra_half_width}
          AND g.dec BETWEEN {dec - half_height} AND {dec + half_height}
          AND g.source_id != {gaia_id}
        ORDER BY tm.j_m ASC
        """

    results = launch_job_async_with_retry(adql)

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

DEFAULT_DB_PATH = '/gaia_database/gaia_dr3_unified_16jcut.db'
GAIA_DR3_EPOCH = 2016.0


def get_target_properties_local(gaia_id, config):
    """
    Retrieve properties of the target star from the local Gaia DR3 SQLite
    database, looked up by DR2 source_id.

    Attempts to use j_m directly. Falls back to estimating J from
    phot_g_mean_mag + teff_gspphot if j_m is NULL. Fails hard if neither
    is available.

    Args:
        gaia_id: Gaia DR2 source_id of the target star (int or str)

    Returns:
        dict with keys: ra, dec, pmra, pmdec, ref_epoch, j_m, teff_val,
                        j_estimated

    Raises:
        ValueError: If target not found, or J magnitude cannot be determined
    """
    db_path = os.environ.get('GAIADATABASEPATH', DEFAULT_DB_PATH)
    gaia_id_str = str(gaia_id)

    # Determine which declination shard to query by doing a full-sky check.
    # We don't know dec yet, so query all shards until the source is found.
    # In practice the target is nearly always in one shard so this is fast.
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        shards = [row[0] for row in cursor.fetchall()]

        row = None
        for shard in shards:
            try:
                cursor = conn.execute(
                    f"SELECT ra, dec, pmra, pmdec, phot_g_mean_mag, "
                    f"teff_gspphot, j_m FROM '{shard}' "
                    f"WHERE dr2_source_id = ?",
                    (gaia_id_str,)
                )
                result = cursor.fetchone()
                if result is not None:
                    row = result
                    break
            except Exception:
                continue
    finally:
        conn.close()

    if row is None:
        raise ValueError(
            f"Target star {gaia_id} not found in local Gaia database"
        )

    ra, dec, pmra, pmdec, g_mag, teff, j_m = row
    pmra  = float(pmra)  if pmra  is not None else 0.0
    pmdec = float(pmdec) if pmdec is not None else 0.0
    j_estimated = False

    TEFF_FALLBACK = 3000.0

    if j_m is not None and np.isfinite(float(j_m)):
        j_m = float(j_m)
        logging.info(f"Target {gaia_id}: using database J={j_m:.3f}")
    else:
        if teff is None or not np.isfinite(float(teff)):
            raise ValueError(
                f"Target star {gaia_id} has no J magnitude and no Teff in "
                f"local database. Cannot determine J magnitude."
            )
        if g_mag is None or not np.isfinite(float(g_mag)):
            raise ValueError(
                f"Target star {gaia_id} has no J magnitude and no G magnitude "
                f"in local database. Cannot determine J magnitude."
            )
        j_m = estimate_j_from_g_teff(float(g_mag), float(teff))
        j_estimated = True
        logging.warning(
            f"Target {gaia_id}: no J magnitude in database. "
            f"Estimated J={j_m:.3f} from G={float(g_mag):.3f}, "
            f"Teff={float(teff):.0f}K. Precision prediction will be approximate."
        )

    if teff is None:
        teff_fallback = float(config.get('teff_fallback', 3000.0))
        logging.warning(
            f"Target {gaia_id}: no Teff in local database. "
            f"Using fallback Teff={teff_fallback:.0f}K. "
            f"Precision prediction will be approximate."
        )
        teff = teff_fallback

    return {
        'ra': float(ra),
        'dec': float(dec),
        'pmra': pmra,
        'pmdec': pmdec,
        'ref_epoch': GAIA_DR3_EPOCH,
        'j_m': j_m,
        'teff_val': float(teff),
        'j_estimated': j_estimated
    }


def get_field_jmag_local(gaia_id, config, expansion_factor=1.0):
    """
    Query the local Gaia DR3 SQLite database for all comparison stars in
    the field surrounding the target, returning J-band magnitudes.

    Stars with no J magnitude are silently excluded (consistent with the
    INNER JOIN behaviour of the archive query).

    Args:
        gaia_id: Gaia DR2 source_id of the target star
        config: Configuration dictionary
        expansion_factor: Factor to expand FOV (1.0 = config FOV, >1.0 = larger)

    Returns:
        Astropy Table with columns: source_id, ra, dec, j_m, teff_val
    """
    import sqlite3
    from astropy.table import Table

    db_path = os.environ.get('GAIADATABASEPATH', DEFAULT_DB_PATH)

    target = get_target_properties_local(gaia_id, config)
    ra, dec = target['ra'], target['dec']

    width  = config['field_of_view']['width_arcmin']  / 60 * expansion_factor
    height = config['field_of_view']['height_arcmin'] / 60 * expansion_factor
    half_width  = width  / 2.0
    half_height = height / 2.0
    ra_half_width = half_width / abs(np.cos(np.radians(dec)))

    min_ra  = ra  - ra_half_width
    max_ra  = ra  + ra_half_width
    min_dec = max(dec - half_height, -90.0)
    max_dec = min(dec + half_height,  90.0)

    # Iterate over all declination shards that overlap the search box
    dec_floors = np.arange(np.floor(min_dec), np.ceil(max_dec) + 1, 1)
    gaia_id_str = str(gaia_id)

    conn = sqlite3.connect(db_path)
    rows = []
    for i in range(len(dec_floors) - 1):
        shard = f"{int(dec_floors[i])}_{int(dec_floors[i + 1])}"
        query = (
            f"SELECT source_id, ra, dec, pmra, pmdec, j_m, teff_gspphot "
            f"FROM '{shard}' "
            f"WHERE dec BETWEEN {min_dec} AND {max_dec} "
            f"AND ra BETWEEN {min_ra} AND {max_ra} "
            f"AND j_m IS NOT NULL "
            f"AND dr2_source_id != ?"
        )
        try:
            cursor = conn.execute(query, (gaia_id_str,))
            rows.extend(cursor.fetchall())
        except Exception as e:
            logging.warning(f"Local DB query error for shard {shard}: {e}")
    conn.close()

    if not rows:
        raise ValueError(
            f"No comparison stars with J magnitudes found in local database "
            f"for field RA={ra:.3f}, Dec={dec:.3f}"
        )

    # Propagate positions to current epoch
    current_epoch = get_current_julian_year()
    delta_t = current_epoch - GAIA_DR3_EPOCH

    source_ids, ras, decs, j_mags, teffs = [], [], [], [], []
    for source_id, r, d, pmra, pmdec, j_m, teff in rows:
        try:
            ra_corr, dec_corr = propagate_position(
                r, d,
                float(pmra)  if pmra  is not None else 0.0,
                float(pmdec) if pmdec is not None else 0.0,
                GAIA_DR3_EPOCH,
                current_epoch
            )
        except Exception:
            ra_corr, dec_corr = r, d
        source_ids.append(source_id)
        ras.append(ra_corr)
        decs.append(dec_corr)
        j_mags.append(float(j_m))
        teffs.append(float(teff) if teff is not None else np.nan)

    table = Table()
    table['source_id'] = source_ids
    table['ra']        = ras
    table['dec']       = decs
    table['j_m']       = j_mags
    table['teff_val']  = teffs

    # Sort by j_m ascending, matching archive query behaviour
    table.sort('j_m')
    return table

def get_target_properties(gaia_id, config=None, use_local_db=True):
    """
    Dispatcher: retrieve target properties from the local database (default)
    or the Gaia TAP archive.
    """
    if use_local_db:
        return get_target_properties_local(gaia_id, config or {})
    else:
        return _get_target_properties_archive(gaia_id)


def get_field_jmag(gaia_id, config, expansion_factor=1.0, use_local_db=True):
    """
    Dispatcher: retrieve field J magnitudes from the local database (default)
    or the Gaia TAP archive.
    """
    if use_local_db:
        return get_field_jmag_local(gaia_id, config, expansion_factor)
    else:
        return _get_field_jmag_archive(gaia_id, config, expansion_factor)