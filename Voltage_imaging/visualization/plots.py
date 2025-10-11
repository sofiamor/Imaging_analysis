import matplotlib.pyplot as plt
import os
from data_io.config import SAVE_DIR

# Ensure SAVE_DIR exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def plot_sync(ephys_trace, frame_times, save=True, fname="sync_plot.png"):
    """
    Plot synchronized electrophysiology trace with frame times as vertical lines.

    Parameters
    ----------
    ephys_trace : np.ndarray
        Synchronized electrophysiology trace
    frame_times : np.ndarray
        Time of each frame (s)
    save : bool
        If True, saves figure to SAVE_DIR
    fname : str
        Filename to save
    """
    plt.figure(figsize=(10,4))
    plt.plot(ephys_trace, color='black', label="Ephys trace")
    for ft in frame_times:
        plt.axvline(ft, color='cyan', alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title("Ephys trace with imaging frame times")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(SAVE_DIR, fname))
    plt.show()


def plot_traces(roi_traces, ephys_trace, save=True, fname="roi_vs_ephys.png"):
    """
    Plot ROI traces overlaid with synchronized ephys trace.

    Parameters
    ----------
    roi_traces : list of np.ndarray
        List of ROI traces (already extracted & synchronized)
    ephys_trace : np.ndarray
        Synchronized ephys trace
    save : bool
        If True, saves figure
    fname : str
        Filename to save
    """
    plt.figure(figsize=(12,5))

    # Plot each ROI
    for i, trace in enumerate(roi_traces):
        plt.plot(trace, label=f"ROI {i+1}", alpha=0.7)

    # Overlay ephys (scaled for visualization)
    ephys_scaled = ephys_trace / max(abs(ephys_trace)) * max([max(t) for t in roi_traces])
    plt.plot(ephys_scaled, color='black', lw=1.5, label="Ephys (scaled)")

    plt.xlabel("Time (frames)")
    plt.ylabel("Signal (ΔF/F or norm.)")
    plt.title("ROI traces vs. Ephys")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(SAVE_DIR, fname))
    plt.show()

import numpy as np
from scipy.signal import correlate

def plot_dff_vs_vm(roi_traces, ephys_trace, save=True, fname="dff_vs_vm.png"):
    """
    Plot ΔF/F (ROI) versus ephys Vm for each ROI.
    Assumes ROI traces are already normalized or ΔF/F.
    """
    plt.figure(figsize=(6,6))
    for i, trace in enumerate(roi_traces):
        plt.scatter(ephys_trace, trace, s=5, alpha=0.7, label=f"ROI {i+1}")
    plt.xlabel("Ephys Vm (mV, or normalized)")
    plt.ylabel("ROI ΔF/F (normalized)")
    plt.title("ΔF/F vs Vm")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(SAVE_DIR, fname))
    plt.show()


def compute_cross_correlation(roi_trace, ephys_trace, fps, ephys_rate):
    """
    Compute cross-correlation between a single ROI trace and ephys trace.
    Returns lag in seconds corresponding to max correlation.
    """
    # Resample ephys to ROI length
    ephys_resampled = np.interp(np.linspace(0, len(ephys_trace), len(roi_trace)),
                                np.arange(len(ephys_trace)), ephys_trace)
    
    # Remove mean
    roi_norm = roi_trace - np.mean(roi_trace)
    ephys_norm = ephys_resampled - np.mean(ephys_resampled)
    
    corr = correlate(roi_norm, ephys_norm, mode='full')
    lags = np.arange(-len(roi_norm)+1, len(roi_norm))
    lag_sec = lags[np.argmax(corr)] / fps  # in seconds
    max_corr = np.max(corr) / (np.std(roi_norm)*np.std(ephys_norm)*len(roi_norm))
    return lag_sec, max_corr


def plot_lag_histogram(roi_traces, ephys_trace, fps, ephys_rate, save=True, fname="lag_histogram.png"):
    """
    Compute lag for each ROI and plot histogram.
    """
    lags = []
    for trace in roi_traces:
        lag_sec, _ = compute_cross_correlation(trace, ephys_trace, fps, ephys_rate)
        lags.append(lag_sec)
    
    plt.figure(figsize=(6,4))
    plt.hist(lags, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel("Lag (s)")
    plt.ylabel("Number of ROIs")
    plt.title("Distribution of ROI vs Ephys lag")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, fname))
    plt.show()
