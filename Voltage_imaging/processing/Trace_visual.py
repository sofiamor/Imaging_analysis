import sys
import os
os.environ['MPLBACKEND'] = 'Qt5Agg'  # Set backend before importing matplotlib
import matplotlib
matplotlib.use('Qt5Agg')  # Explicitly set backend
import matplotlib.pyplot as plt
import os
import numpy as np
from pyabf import abf
import scipy.signal as signal
from scipy.signal import find_peaks, welch, iirnotch, butter, filtfilt
from scipy.stats import linregress
import datetime
from matplotlib.ticker import MaxNLocator

print("Current Directory:", os.getcwd())
project_path = r"C:\Users\sofik\.vscode\Voltage_imaging"
print("project_path:", project_path)
sys.path.append(project_path)

# Check if functions.py exists at this location
if os.path.isfile(os.path.join(project_path, "read_abf.py")):
    print("functions.py found at:", project_path)
else:
    print("functions.py NOT found at:", project_path)

# Now import Abfdata from functions
import data_io.read_abf
print(sys.path)

# ---------- Processing function ----------
def process_abf_data(file_path, color, label, window=(0, 50)):
    # Load the data
    data = read_abf.Abfdata(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Extract trace and time data
    trace_data = data.extract_trace_data()
    time_values = data.get_time_values() * 1000  # convert to ms

    excluded_indices = [7]  # exclude first 6 sweeps
    filtered_indices = [idx for idx in range(len(trace_data)) if idx not in excluded_indices]

    # Create a Gaussian window for filtering
    std_dev = 5
    window_size = 10
    window_g = signal.windows.gaussian(window_size, std_dev)
    filtered_trace_data = trace_data[filtered_indices]

    # Extract stimulation times
    pulse_data = [data.extract_pulse_data() for _ in filtered_trace_data]
    stim_time = []
    for pulse in pulse_data[0:1]:
        peaks, _ = signal.find_peaks(pulse, height=2.1)
        stim_times = time_values[peaks]
        stim_time.append(stim_times)
        pulse_width = 2  # ms
        for stim in stim_times:
            plt.vlines([stim, stim + pulse_width], ymin=-200, ymax=200,
                       color='paleturquoise', linewidth=.2)

    # Filter sweeps
    filt_trace_data = []
    for sweep_data in filtered_trace_data:
        filtered_sweep_data = signal.convolve(sweep_data, window_g, mode='same') / sum(window_g)
        filt_trace_data.append(filtered_sweep_data)

    # Average sweeps and baseline correction
    averaged_data = data.average_abf_sweeps()
    window_g2 = signal.windows.gaussian(5, 2)
    averaged_data_data = signal.convolve(averaged_data, window_g2, mode='same') / sum(window_g2)
    baseline_averageddata = data.baseline_correction(averaged_data_data)

    # Additional notch/lowpass filtering if desired
    fs = 20000
    f, Pxx = welch(baseline_averageddata, fs, nperseg=2048)
    f0 = 60
    Q = 30
    b, a = iirnotch(f0, Q, fs)
    cleaned_trace = filtfilt(b, a, baseline_averageddata)

    def butter_lowpass(cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    cutoff = 50
    b, a = butter_lowpass(cutoff, fs)
    smoothed_trace = filtfilt(b, a, cleaned_trace)

    # Plot averaged trace
    plt.plot(time_values, baseline_averageddata, label=f'{label}', color=color, alpha=0.8)

    # ---- NEW: quantify events ----
    amps, slps = measure_event_metrics(time_values,
                                       baseline_averageddata,
                                       stim_times=stim_time[0],
                                       window=window)  # custom window
    return amps, slps


# ---------- Collect data ----------
plt.figure(figsize=(15, 4))
all_amp_means, all_slope_means, cond_labels = [], [], []
measurement_window = (300, 310)  # <-- change this to adjust post-stimulus window (ms)

# ---------- Trace plot ----------
plt.xlim(250, 2000)
plt.ylim(-5, 50)
plt.plot([measurement_window[0], measurement_window[0]], [-100, 200], 'k--', linewidth=0.8)
plt.plot([measurement_window[1], measurement_window[1]], [-100, 200], 'k--', linewidth=0.8)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=15))
plt.xlabel('Time (ms)', color='black')
plt.ylabel('Current (pA)', color='black')
plt.xticks(color='black')
plt.yticks(color='black')
plt.legend()
plt.tight_layout()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#trace_pdf = f"Z:/smorou/Analysis/ephys_analysis/gabagluts/{base_name}_{timestamp}_traces.pdf"
#plt.savefig(trace_pdf, transparent=True, format="pdf")
#print(f"Trace plot saved as: {trace_pdf}")
plt.show()