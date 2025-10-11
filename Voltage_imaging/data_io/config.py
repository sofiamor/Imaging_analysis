# config.py

# General
SAVE_DIR = r"C:\Users\sofik\Desktop\voltage_imaging_analysisFR"
FIG_FORMAT = "pdf"

# Imaging
FPS = 1000  # imaging frame rate in Hz
ROI_METHOD = "manual"  # "manual" or "auto"
ROI_RADIUS = 5  # for circular ROIs

# Ephys
EPHYS_SAMPLING_RATE = 50000  # Hz

# Synchronization
SYNC_METHOD = "frame_trigger"  # or "cross_correlation"
OFFSET_MS = 0  # manual correction if needed

# Plotting
TIME_LIMIT = (0, 2000)  # ms, x-axis limits
Y_LIMIT_IMAGING = (-100, 100)  # ΔF/F %
Y_LIMIT_EPHYS = (-200, 50)  # pA or mV
