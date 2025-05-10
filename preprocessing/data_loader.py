import os
import numpy as np


def load_raw_data(path: str) -> np.ndarray:
    """
    Load raw multimodal data stored as a NumPy .npy file or directory of .npy files.
    If path is a directory, loads all .npy files and stacks them.
    Returns array of shape (n_rois, time_points).
    """
    if os.path.isdir(path):
        arrays = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith('.npy'):
                arrays.append(np.load(os.path.join(path, fname)))
        if not arrays:
            raise FileNotFoundError(f"No .npy files found in directory {path}")
        data = np.stack(arrays, axis=0)
    else:
        data = np.load(path)
    return data


def save_processed_data(data: np.ndarray, path: str) -> None:
    """
    Save processed data as a NumPy .npy file.
    """
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    np.save(path, data)