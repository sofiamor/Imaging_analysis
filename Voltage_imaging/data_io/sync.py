import sys, os
# Set your project directory
project_path = r"C:\Users\sofik\.vscode\Voltage_imaging"
# Change working directory
os.chdir(project_path)

# Ensure Python knows where to look for your modules
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import numpy as np
from data_io.config import *

def synchronize(abf_data, abf_time, abf_rate, frame_times, method="resample"):
    """
    Synchronize imaging frames with ephys trace.

    Parameters
    ----------
    abf_data : np.ndarray
        Electrophysiology data.
    abf_time : np.ndarray
        Time vector for ephys (seconds).
    abf_rate : float
        Sampling rate of ABF data (Hz).
    frame_times : np.ndarray
        Time stamps of each imaging frame (seconds).
    method : str
        "resample" = interpolate ABF onto frame times
        "nearest" = nearest neighbor

    Returns
    -------
    aligned_abf : np.ndarray
        ABF signal resampled to imaging frames.
    """
    abf_rate = float(config.EPHYS_SAMPLING_RATE)
    
    if method == "resample":
        aligned_abf = np.interp(frame_times, abf_time, abf_data)
    elif method == "nearest":
        idx = np.searchsorted(abf_time, frame_times)
        idx[idx >= len(abf_data)] = len(abf_data) - 1
        aligned_abf = abf_data[idx]
    else:
        raise ValueError("Unknown sync method.")

    return aligned_abf
