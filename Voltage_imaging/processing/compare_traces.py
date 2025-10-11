# processing/compare_traces.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

def align_and_compare(im_trace, ephys_trace, fps, ephys_rate,
                      start_time=0, end_time=None, smoothing=False):
    """
    Synchronize and compare imaging and ephys traces.
    """

    # Time vectors
    t_im = np.arange(len(im_trace)) / fps
    t_ephys = np.arange(len(ephys_trace)) / ephys_rate

    # Restrict to time window
    if end_time is None:
        end_time = min(t_im[-1], t_ephys[-1])
    mask_im = (t_im >= start_time) & (t_im <= end_time)
    mask_ephys = (t_ephys >= start_time) & (t_ephys <= end_time)
    im_trace = im_trace[mask_im]
    ephys_trace = ephys_trace[mask_ephys]
    t_im = t_im[mask_im]

    # Normalize
    im_norm = (im_trace - np.mean(im_trace)) / np.std(im_trace)
    ephys_norm = (ephys_trace - np.mean(ephys_trace)) / np.std(ephys_trace)

    # Resample ephys to match imaging FPS
    ephys_resampled = resample(ephys_norm, len(im_norm))

    # Optional smoothing
    if smoothing:
        from scipy.ndimage import gaussian_filter1d
        im_norm = gaussian_filter1d(im_norm, 2)
        ephys_resampled = gaussian_filter1d(ephys_resampled, 2)

    # Compute correlation
    corr = np.corrcoef(im_norm, ephys_resampled)[0, 1]

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(t_im, im_norm, label="Imaging (norm.)", alpha=0.7)
    plt.plot(t_im, ephys_resampled, label="Ephys (resampled, norm.)", alpha=0.7)
    plt.title(f"Imaging vs. Ephys comparison (r = {corr:.2f})")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Signal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return corr, im_norm, ephys_resampled
