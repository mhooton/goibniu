import numpy as np
import joblib
from paths import MODEL_PATH
import warnings
warnings.filterwarnings('ignore', message='Trying to unpickle estimator')

def convert_j_to_zyj(jmag, config):
    """
    Convert J-band magnitude to zYJ filter magnitude using linear transformation.

    Args:
        jmag: J-band magnitude(s)
        config: Configuration dictionary

    Returns:
        zYJ magnitude(s)
    """
    slope = config['j_to_zyj_conversion']['slope']
    intercept = config['j_to_zyj_conversion']['intercept']
    return (jmag - intercept) / slope


def combined_mag(mag_array):
    """
    Calculate combined magnitude from multiple stars by adding their fluxes.

    Args:
        mag_array: Array of magnitudes

    Returns:
        Combined magnitude (scalar)
    """
    # Convert magnitudes to fluxes and sum
    artificial_flux = np.sum(10 ** (-0.4 * mag_array))
    # Convert back to magnitude
    artificial_mag = - 2.5 * np.log10(artificial_flux)
    return artificial_mag


def effective_mg(mag_array, target_index):
    """
    Calculate the effective magnitude representing the precision limit
    of differential photometry.

    Parameters:
    -----------
    mag_array : array-like
        Array of magnitudes (target + comparison stars)
    target_index : int
        Index of the target star in mag_array

    Returns:
    --------
    float
        Effective magnitude representing the combined noise floor
    """

    # Extract target magnitude
    m_target = mag_array[target_index]

    # Get all comparison star magnitudes (excluding target)
    comp_mags = np.delete(mag_array, target_index)

    # Convert comparison magnitudes to flux and sum to get artificial star flux
    comp_fluxes = 10 ** (-0.4 * comp_mags)
    artificial_flux = np.sum(comp_fluxes)

    # Convert artificial flux back to magnitude
    m_artificial = -2.5 * np.log10(artificial_flux)

    # Calculate effective differential magnitude
    print(m_target)
    print(comp_mags)
    print(m_artificial)
    m_diff = 2.5 * np.log10(10 ** (0.4 * m_target) + 10 ** (0.4 * m_artificial))

    return m_diff


def prediction_from_fit(total_mag, config):
    """
    Predict photometric precision using quadratic fit coefficients.

    Args:
        total_mag: Combined magnitude of field
        config: Configuration dictionary

    Returns:
        Predicted precision (log10 scale)
    """
    coeffs = config['quadratic_fit_coefficients']
    return np.polyval(coeffs, total_mag)  # Precision in log10

def prediction_from_DT(features):
    """
    Predict photometric precision using trained Decision Tree model.

    Args:
        features: DataFrame with columns ['Comp stars', 'zYJ mag', 'Combined mag', 'Teff']

    Returns:
        Predicted precision (linear scale)
    """
    # OLD:
    # model_dict = joblib.load('precision_prediction_model.joblib')

    # NEW:
    model_dict = joblib.load(MODEL_PATH)

    model = model_dict['model']
    feature_array = features[['Comp stars', 'zYJ mag', 'Combined mag', 'Teff']].values
    precision = model.predict(feature_array)[0]
    return precision
