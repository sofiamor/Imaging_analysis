import tifffile
import numpy as np

def load_tif(file_path, fps):
    """
    Load imaging stack (TIFF).

    Parameters
    ----------
    file_path : str
        Path to the TIFF stack.
    fps : float
        Imaging frame rate (Hz).

    Returns
    -------
    stack : np.ndarray
        Image stack (frames, height, width).
    frame_times : np.ndarray
        Time vector for each frame (seconds).
    """
    stack = tifffile.imread(file_path)  # shape: (frames, h, w)
    n_frames = stack.shape[0]
    frame_times = np.arange(n_frames) / fps

    return stack, frame_times
