import numpy as np
import matplotlib.pyplot as plt

def extract_high_points(trace, threshold=None, top_percent=5):
    """
    Identify high-count points in a trace.
    
    Parameters
    ----------
    trace : np.ndarray
        1D signal (ROI or ephys)
    threshold : float or None
        Absolute value threshold. If None, uses top_percent percentile.
    top_percent : float
        Percentage of points to consider as high-count if threshold is None
    
    Returns
    -------
    indices : np.ndarray
        Indices of the high-count points
    """
    if threshold is None:
        threshold = np.percentile(trace, 100 - top_percent)
    indices = np.where(trace >= threshold)[0]
    return indices


def extract_subthreshold_segments(trace, indices, window_before=10, window_after=10):
    """
    Extract subthreshold segments around selected indices.
    
    Parameters
    ----------
    trace : np.ndarray
        1D signal
    indices : array-like
        Points to center the extraction window
    window_before : int
        Number of samples to include before the point
    window_after : int
        Number of samples to include after the point
        
    Returns
    -------
    segments : np.ndarray
        Array of shape (len(indices), window_before + window_after)
    """
    segments = []
    n = len(trace)
    for idx in indices:
        start = max(idx - window_before, 0)
        end = min(idx + window_after, n)
        segment = trace[start:end]
        # If segment shorter than window, pad with NaN for alignment
        if len(segment) < window_before + window_after:
            segment = np.pad(segment, (0, window_before + window_after - len(segment)), 
                             mode='constant', constant_values=np.nan)
        segments.append(segment)
    return np.array(segments)


def average_segments(segments):
    """
    Compute the average of subthreshold segments, ignoring NaNs.
    """
    return np.nanmean(segments, axis=0)


def plot_average_trace(avg_trace, window_before=10, window_after=10, save=True, fname="avg_subthreshold.png"):
    """
    Plot averaged subthreshold trace.
    
    Parameters
    ----------
    avg_trace : np.ndarray
        Averaged trace
    window_before, window_after : int
        Number of samples before/after the high point
    save : bool
        If True, saves figure
    fname : str
        Filename
    """
    plt.figure(figsize=(6,4))
    t = np.arange(-window_before, window_after)
    plt.plot(t, avg_trace, color='red', lw=2)
    plt.xlabel("Samples relative to peak")
    plt.ylabel("Signal (ΔF/F or normalized)")
    plt.title("Average Subthreshold Trace")
    plt.axvline(0, color='black', ls='--', lw=1)
    plt.tight_layout()
    if save:
        plt.savefig(fname)
    plt.show()
