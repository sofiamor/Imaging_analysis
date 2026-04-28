import matplotlib.pyplot as plt
import os
#from data_io.config import SAVE_DIR
import numpy as np
from scipy.signal import correlate

# Ensure SAVE_DIR exists
#if not os.path.exists(SAVE_DIR):
#    os.makedirs(SAVE_DIR)

def plot_metrics(metrics):
    x = np.arange(1, len(metrics["Signal"]) + 1)

    plt.figure(figsize=(7, 9))
    plt.subplot(3, 1, 1)
    plt.title("Signal / Noise / Camera-corrected Baseline")
    plt.scatter(x, metrics["Signal"], label="Signal")
    plt.scatter(x, metrics["Noise"], label="Noise")
    #plt.scatter(x, metrics["Baseline_Cameracorr"], label="BaselineCameraCorr")
    plt.plot(x, metrics["Signal"])
    plt.plot(x, metrics["Noise"])
    #plt.plot(x, metrics["Baseline_Cameracorr"])
    plt.xlabel("Trace #")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.title("dF/F")
    plt.scatter(x, metrics["dFF"])
    plt.plot(x, metrics["dFF"])
    plt.xlabel("Trace #")
    plt.ylabel("dF/F")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.title("SNR")
    plt.scatter(x, metrics["SNR"])
    plt.plot(x, metrics["SNR"])
    plt.xlabel("Trace #")
    plt.ylabel("SNR")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

