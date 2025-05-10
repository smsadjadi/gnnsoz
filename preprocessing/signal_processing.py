import numpy as np


def preprocess_fmri(raw_fmri: np.ndarray) -> np.ndarray:
    """
    Basic preprocessing: detrend (demean) and z-score each ROI time series.
    Args:
        raw_fmri: array shape (n_rois, time_points)
    Returns:
        processed array same shape
    """
    # Demean
    demeaned = raw_fmri - np.mean(raw_fmri, axis=1, keepdims=True)
    # Z-score
    std = np.std(demeaned, axis=1, keepdims=True)
    std[std == 0] = 1.0
    processed = demeaned / std
    return processed


def preprocess_eeg(raw_eeg: np.ndarray) -> np.ndarray:
    """
    Basic preprocessing: bandpass filter simulation: demean and scale.
    Args:
        raw_eeg: array shape (n_channels, time_points)
    Returns:
        processed array same shape
    """
    # Demean
    demeaned = raw_eeg - np.mean(raw_eeg, axis=1, keepdims=True)
    # Scale to unit variance
    std = np.std(demeaned, axis=1, keepdims=True)
    std[std == 0] = 1.0
    processed = demeaned / std
    return processed


def preprocess_dmri(raw_dmri: str) -> dict:
    """
    Load a precomputed diffusion connectivity matrix stored as .npy
    and return as a dict.
    Args:
        raw_dmri: path to .npy file of shape (n_rois, n_rois)
    Returns:
        {'connectivity': np.ndarray}
    """
    import numpy as np
    conn = np.load(raw_dmri)
    return {'connectivity': conn}