import numpy as np

def synchronize_flexible(trace_data, trace_time, frame_times, method="interp"):
    """
    Synchronize electrophysiological trace to imaging frames flexibly.

    Parameters
    ----------
    trace_data : array-like
        1D array of ephys values.
    trace_time : array-like
        1D array of ephys time points (same length as trace_data).
    frame_times : array-like
        1D array of frame timestamps from imaging.
    method : str
        'interp' (default) uses linear interpolation,
        'nearest' uses nearest-neighbor matching.

    Returns
    -------
    aligned_trace : np.ndarray
        Trace values resampled to frame_times.
    valid_frame_times : np.ndarray
        Frame times within the overlapping time window.
    """

    trace_data = np.ravel(trace_data)
    trace_time = np.ravel(trace_time)
    frame_times = np.ravel(frame_times)

    # Ensure overlap between trace and frames
    t_min = max(trace_time[0], frame_times[0])
    t_max = min(trace_time[-1], frame_times[-1])

    mask = (frame_times >= t_min) & (frame_times <= t_max)
    valid_frames = frame_times[mask]

    if len(valid_frames) == 0:
        raise ValueError("No overlapping time between ephys and imaging data.")

    if method == "interp":
        aligned_trace = np.interp(valid_frames, trace_time, trace_data)
    elif method == "nearest":
        idx = np.searchsorted(trace_time, valid_frames)
        idx = np.clip(idx, 0, len(trace_time) - 1)
        aligned_trace = trace_data[idx]
    else:
        raise ValueError("Unknown method: choose 'interp' or 'nearest'.")

    return aligned_trace, valid_frames
