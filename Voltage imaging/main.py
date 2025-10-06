import sys, os

print("Current Directory:", os.getcwd())
config_path = r"sofiamor/Imaging_analysis/Voltage imaging/io"
print("functions_path:", config_path)
sys.path.append(config_path)

# Check if functions.py exists at this location
if os.path.isfile(os.path.join(config_path, "config.py")):
    print("config.py found at:", config_path)
else:
    print("config.py NOT found at:", config_path)

# Now import Abfdata from functions
from config import *
from io.read_abf import load_abf
from io.read_tif import load_tif
from io.sync import synchronize
from processing.roi import select_rois
from processing.extract_traces import extract_signals
from processing.compare import compare_signals
from visualization.plots import plot_sync, plot_traces
print(sys.path)

def main():
    # 1. Load data
    abf_data, abf_time = load_abf("data/example.abf", EPHYS_SAMPLING_RATE)
    tiff_stack, frame_times = load_tif("data/example.tif", FPS)

    # 2. Synchronize
    aligned_abf, aligned_frames = synchronize(abf_data, abf_time, frame_times)

    # 3. ROI Selection
    rois = select_rois(stack, method="ask", roi_radius=5, predefined_coords=[(50, 50), (120, 80)])

    # 4. Extract ROI signals
    roi_traces = extract_signals(tiff_stack, rois)

    # 5. Compare with Ephys
    compare_results = compare_signals(roi_traces, aligned_abf)

    # 6. Plot
    plot_sync(aligned_abf, aligned_frames)
    plot_traces(roi_traces, aligned_abf)

    print("Pipeline finished successfully. Results saved to", SAVE_DIR)

if __name__ == "__main__":
    main()
