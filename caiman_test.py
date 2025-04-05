import numpy as np
import tifffile
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf
from caiman.source_extraction.cnmf import deconvolution
import matplotlib.pyplot as plt
import os
import tifffile as tf


# Define paths
data_path = f'F:/histed/histed_2101_test2_n2_200msSTIM/histed_2101_test2_n2_200msSTIM_MMStack.ome.tif'
tmp_dir = "C:/Users/LabPC/Downloads/caiman_tmp/"
os.makedirs(tmp_dir, exist_ok=True)

# Load OME-TIFF safely
print("Loading OME-TIFF...")
try:
    data = tf.imread(data_path, is_ome=False)  # 🔥 Force regular TIFF mode
except Exception as e:
    print(f"❌ Error loading OME-TIFF: {e}")
    exit()

print(f"Loaded data shape: {data.shape}")

# Ensure it's in the correct shape (frames, height, width)
data = data[:4749, :417, :816]  # Crop if necessary
data = np.array(data, dtype=np.float32)  # Convert to float32

# Fix CaImAn mmap filename issue
file_name_without_ext = os.path.basename(data_path)
mmap_path = os.path.join(tmp_dir, file_name_without_ext)
print(f"Saving mmap to: {mmap_path}")

# Convert to CaImAn movie
caiman_movie = cm.movie(data)

# Save as mmap (CaImAn will add metadata)
caiman_movie.save(mmap_path, order='C')

# Find the actual mmap filename
mmap_file = None
for file in os.listdir(tmp_dir):
    if file.startswith(file_name_without_ext) and file.endswith(".mmap"):
        mmap_file = os.path.join(tmp_dir, file)
        break

if mmap_file:
    print(f"✅ Found mmap file: {mmap_file}")
else:
    print("❌ ERROR: mmap file not found!")

# Load the correct mmap file
if mmap_file:
    mmap_movie = cm.load(mmap_file)
    print("✅ mmap loaded successfully!")



# Motion correction
mc = MotionCorrect([mmap_file], max_shifts=(6, 6), strides=(48, 48), overlaps=(24, 24))
mc.motion_correct(save_movie=True)

# Reload corrected movie
motion_corrected_data = cm.load(mc.mmap_file)

# Ensure mc.mmap_file is set properly
if mc.mmap_file is None:
    print("Error: mc.mmap_file is None. Motion correction may have failed.")
    exit()

# Check for NaNs in the motion-corrected data
if np.isnan(motion_corrected_data).any():
    print("Warning: NaN values detected. Replacing NaNs with the mean value.")
    motion_corrected_data = np.nan_to_num(motion_corrected_data, nan=np.nanmean(motion_corrected_data))

# Source extraction
cnm = cnmf.CNMF(n_processes=100, k=10, gSig=(4, 4), merge_thresh=0.8)
cnm.fit(motion_corrected_data)

print(f"Shape of cnm.estimates.C: {cnm.estimates.C.shape}")

if cnm.estimates.C is None:
    print("Error: cnm.estimates.C is None!")
elif cnm.estimates.C.size == 0:
    print("Error: cnm.estimates.C is empty!")

p = 2  # Default order for AR model
sn = np.nanstd(cnm.estimates.C, axis=1)  # Estimate noise level
g = [0.9] * p  # Assume some reasonable autoregressive parameters

#C_dec, S_dec, b_dec, c_dec, g_dec = deconvolution.constrained_foopsi(
#    cnm.estimates.C, p=p, sn=sn, g=g
#)

print("Shape of C:", cnm.estimates.C.shape)
print("Any NaNs in C?", np.isnan(cnm.estimates.C).any())
print("Any infinities in C?", np.isinf(cnm.estimates.C).any())
print("C min/max:", np.min(cnm.estimates.C), np.max(cnm.estimates.C))
print("Shape of sn:", sn.shape)
print("Any NaNs in sn?", np.isnan(sn).any())
print("Any negative values in sn?", (sn < 0).any())
print("g:", g)
print("Any NaNs in g?", np.isnan(g).any())
print("Any negative values in g?", (np.array(g) < 0).any())

print(vars(cnm.params))

# Visualization
plt.figure()
for i in range(min(10, cnm.estimates.C.shape[0])):  # Plot first 10 traces
    plt.plot(cnm.estimates.C[i, :])
plt.xlabel('Time')
plt.ylabel('Fluorescence')
plt.show()
