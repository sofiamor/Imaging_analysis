import sys, os

# Set your project directory
project_path = r"C:\Users\sofik\.vscode\Voltage_imaging"

# Change working directory
os.chdir(project_path)

# Ensure Python knows where to look for your modules
if project_path not in sys.path:
    sys.path.insert(0, project_path)

print("Current Directory:", os.getcwd())
    
#import setup_path 

from data_io.config import *
from data_io.read_abf import load_abf
from data_io.read_tif import load_tif
from data_io.sync import synchronize
from processing.roi import select_rois
from processing.extract_traces import extract_signals
from processing.compare_traces import align_and_compare
from visualization.plots import plot_sync, plot_traces
from processing.averaging_subthreshold import (
    extract_high_points,
    extract_subthreshold_segments,
    average_segments,
    plot_average_trace
)

# Example for one ROI trace
roi = roi_traces[0]

# 1. Find high-count points (top 5%)
high_idx = extract_high_points(roi, top_percent=5)

# 2. Extract segments around these points
segments = extract_subthreshold_segments(roi, high_idx, window_before=20, window_after=20)

# 3. Compute average
avg_trace = average_segments(segments)

# 4. Plot
plot_average_trace(avg_trace, window_before=20, window_after=20)

print(sys.path)

def main():
    # 1. Load data
    abf_data, abf_time = load_abf("data/example.abf", EPHYS_SAMPLING_RATE)
    tiff_stack, frame_times = load_tif("/DESKTOP-818PRUI/Gdata/Sofi/20251004/CELL1/Sensitivity/010-Sens70-2x2/example.tif", FPS)

    # 2. Synchronize
    aligned_abf, aligned_frames = synchronize(abf_data, abf_time, frame_times)

    # 3. ROI Selection
    rois = select_rois(stack, method="ask", roi_radius=5, predefined_coords=[(50, 50), (120, 80)])

    # 4. Extract ROI signals
    roi_traces = extract_signals(tiff_stack, rois)

    # 5. Compare with Ephys
    compare_results = align_and_compare(roi_traces, aligned_abf)

    # 6. Plot
    plot_sync(aligned_abf, aligned_frames)
    plot_traces(roi_traces, aligned_abf)

    print("Pipeline finished successfully. Results saved to", SAVE_DIR)

if __name__ == "__main__":
    main()
