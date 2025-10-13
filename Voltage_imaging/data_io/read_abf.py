import numpy as np
import pyabf, sys
#sys.path.append("C:/Users/sofik/.vscode/Voltage_imaging/")
#import processing.filter

class Abfdata:

    def __init__(self, abf_file_path):
        self.abf = pyabf.ABF(abf_file_path)

    def load_abf_file(self, __init__=None, abf_file_path=None):
        abf = pyabf.ABF(abf_file_path)
        return abf
    

    def extract_trace_data(self, downsample=False):
        # Initialize an array to store sweep data
        trace_data = np.empty((self.abf.sweepCount, self.abf.sweepPointCount))

        # Read data from each sweep
        for sweep_number in range(self.abf.sweepCount):
            self.abf.setSweep(sweep_number)
            trace_data[sweep_number, :] = self.abf.sweepY

        if downsample:
            time_values = np.arange(self.abf.sweepPointCount) * self.abf.dataSecPerPoint
            downsampled_data = [np.interp(time_values, np.linspace(0, time_values[-1], len(sweep_data)), sweep_data) for sweep_data in trace_data]
            return downsampled_data
        else:
            return trace_data

    
    def get_time_values(self):
        time_values = np.arange(self.abf.sweepPointCount) * self.abf.dataSecPerPoint
        return time_values

    def extract_pulse_data(self, pulse_channel=0):
        self.abf.setSweep(sweepNumber=0, channel=pulse_channel)
        pulse_data = self.abf.sweepY
        print(pulse_data)
        return pulse_data

    def downsample_data(self, time_values, data):
        downsampled_data = {}
        downsampled_data = np.interp(time_values, np.linspace(0, time_values[-1], len(data)), data)
        return downsampled_data

    def analyze_pulses(self, num_pulses, first_pulse_offset, last_pulse_offset, normal_offset, pulse, time_values, downsampled_data):
        for i in range(num_pulses):
            if i == 0:
                pulse_offset = first_pulse_offset
            elif i == 9:
                pulse_offset = last_pulse_offset
            else:
                pulse_offset = normal_offset
                pulse_start = i * pulse + first_pulse_offset
            pulse_end = pulse_start + pulse
            pulse_indices = np.where((time_values >= pulse_start) & (time_values < pulse_end))[0]
            pulse_data = downsampled_data[pulse_indices]  - np.mean(downsampled_data[pulse_indices])
            pulse_time = time_values[pulse_indices] - pulse_start  # Relative time within the pulse
            return pulse_data, pulse_time   

    def average_abf_sweeps(self):
        # Initialize an array to store the sum of the sweeps
        sum_sweeps = np.zeros(self.abf.sweepPointCount)

        # Add the data from each sweep to sum_sweeps
        for sweep_number in range(self.abf.sweepCount):
            self.abf.setSweep(sweep_number)
            sum_sweeps += self.abf.sweepY

        # Divide by the number of sweeps to get the average
        average_sweeps = sum_sweeps / self.abf.sweepCount

        return average_sweeps
    
    def baseline_correction(self, trace_data):
        baseline = np.mean(trace_data[:1000])  # calculate baseline as mean of first 100 points
        trace_data_baseline_corrected = trace_data - baseline  # subtract baseline
        return trace_data_baseline_corrected
    
    print('functions.py loaded successfully bitch!')