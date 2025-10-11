def load_abf(file_path, sampling_rate):
    import pyabf
    import numpy as np

    abf = pyabf.ABF(file_path)
    n_channels = abf.channelCount
    n_sweeps = abf.sweepCount  # usually 1 per channel

    # Collect all sweeps for all channels
    data = []
    for ch in range(n_channels):
        abf.setSweep(ch)  # move to channel ch
        data.append(abf.sweepY.copy())  # extract numpy array
    data = np.array(data)  # shape = (n_channels, n_samples)

    time = abf.sweepX.copy()  # same for all channels
    return data, time
