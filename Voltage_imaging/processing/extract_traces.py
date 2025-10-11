import numpy as np

def extract_signals(stack, rois, method="dff", baseline_frames=None):
    """
    Extract ROI signals from an image stack.

    Parameters
    ----------
    stack : np.ndarray
        Image stack (frames, height, width)
    rois : list of dicts
        ROI dictionaries from select_rois()
    method : str
        "raw" = mean signal
        "dff" = ΔF/F
    baseline_frames : list or np.ndarray
        Frames used to calculate baseline F0 (for dF/F)

    Returns
    -------
    signals : dict
        keys: "raw" and/or "dff", values: list of np.ndarray (per ROI)
    """
    signals = {"raw": [], "dff": []}

    for roi in rois:
        mask = roi["mask"]
        # Extract mean signal over ROI pixels
        roi_trace = stack[:, mask].mean(axis=1)
        signals["raw"].append(roi_trace)

        if method.lower() == "dff":
            if baseline_frames is None:
                # default: first 10% of frames
                n_base = max(1, int(stack.shape[0]*0.1))
                baseline = roi_trace[:n_base].mean()
            else:
                baseline = roi_trace[baseline_frames].mean()
            dff = (roi_trace - baseline) / baseline
            signals["dff"].append(dff)

    # If method is only raw, remove empty dff lists
    if method.lower() == "raw":
        signals.pop("dff")
    elif method.lower() == "dff":
        signals.pop("raw")  # optional, keep only dff

    return signals
