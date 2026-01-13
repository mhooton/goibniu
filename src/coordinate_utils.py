import numpy as np
from datetime import datetime
from astropy.time import Time
from astropy.wcs import WCS

def propagate_position(ra, dec, pmra, pmdec, ref_epoch, target_epoch):
    """
    Propagate star position from reference epoch to target epoch using proper motion.

    Args:
        ra: Right ascension at reference epoch (degrees)
        dec: Declination at reference epoch (degrees)
        pmra: Proper motion in RA * cos(dec) (mas/yr)
        pmdec: Proper motion in Dec (mas/yr)
        ref_epoch: Reference epoch (Julian year, e.g., 2015.5)
        target_epoch: Target observation epoch (Julian year)

    Returns:
        ra_new, dec_new: Propagated coordinates (degrees)
    """
    # Handle missing proper motions (treat as zero)
    if pmra is None or np.ma.is_masked(pmra) or not np.isfinite(pmra):
        pmra = 0.0
    if pmdec is None or np.ma.is_masked(pmdec) or not np.isfinite(pmdec):
        pmdec = 0.0

    # Time difference in years
    dt = target_epoch - ref_epoch

    # Convert proper motions from mas/yr to degrees/yr
    pmra_deg = pmra / (3600.0 * 1000.0)  # mas -> degrees
    pmdec_deg = pmdec / (3600.0 * 1000.0)

    # Propagate positions
    # Note: pmra already includes cos(dec) factor in Gaia
    ra_new = ra + pmra_deg * dt
    dec_new = dec + pmdec_deg * dt

    return ra_new, dec_new

def get_current_julian_year():
    """
    Get current date as Julian year.

    Returns:
        Current Julian year (e.g., 2025.789)
    """

    now = datetime.now()
    t = Time(now)
    return t.jyear

def sky_to_pixel(ra, dec, wcs_obj):
    """
    Transform RA/Dec to detector X/Y coordinates.

    Args:
        ra: Right ascension (degrees), scalar or array
        dec: Declination (degrees), scalar or array
        wcs_obj: astropy.wcs.WCS object

    Returns:
        x, y: Pixel coordinates (0-indexed)
    """
    x, y = wcs_obj.world_to_pixel_values(ra, dec)
    return x, y


def pixel_to_sky(x, y, wcs_obj):
    """
    Transform detector X/Y to RA/Dec coordinates.

    Args:
        x: X pixel coordinate (0-indexed), scalar or array
        y: Y pixel coordinate (0-indexed), scalar or array
        wcs_obj: astropy.wcs.WCS object

    Returns:
        ra, dec: Sky coordinates (degrees)
    """
    ra, dec = wcs_obj.pixel_to_world_values(x, y)
    return ra, dec

def create_wcs(target_ra, target_dec, target_x, target_y, config):
    """
    Create WCS object for coordinate transformations.

    Args:
        target_ra: Target RA in degrees
        target_dec: Target Dec in degrees
        target_x: Target X pixel position on detector
        target_y: Target Y pixel position on detector
        config: Configuration dictionary

    Returns:
        astropy.wcs.WCS object
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [target_x, target_y]
    wcs.wcs.crval = [target_ra, target_dec]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.pc = [[config['wcs']['pc1_1'], config['wcs']['pc1_2']],
                  [config['wcs']['pc2_1'], config['wcs']['pc2_2']]]
    wcs.wcs.cdelt = [1.0, 1.0]
    wcs.wcs.cunit = ['deg', 'deg']
    return wcs