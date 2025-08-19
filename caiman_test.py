import numpy as np
import tifffile as tf
import caiman as cm
import os
import pandas as pd
import matplotlib.pyplot as plt
from caiman.source_extraction.cnmf import params as cnmf_params
from caiman.source_extraction.cnmf import CNMF

# Define paths
data_path = 'F://histed//histed_2101_test2_n2_200msSTIM//histed_2101_test2_n2_200msSTIM//roi_histed_2101_test2_n2_200msSTIM_MMStack.ome.tif'
tmp_dir = "C:/Users/LabPC/Downloads/caiman_tmp/"
os.makedirs(tmp_dir, exist_ok=True)

# Load OME-TIFF safely
print("🔄 Loading OME-TIFF...")
try:
    data = tf.imread(data_path, is_ome=False)  # 🔥 Force regular TIFF mode
except Exception as e:
    print(f"❌ Error loading OME-TIFF: {e}")
    exit()

print(f"✅ TIFF loaded! Shape: {data.shape}, dtype: {data.dtype}")

# Ensure 3D shape (frames, height, width)
if data.ndim == 2:
    data = data[np.newaxis, :, :]  # Add time dimension

# Convert to float32
data = np.array(data, dtype=np.float32)  # Convert to float32

# Fix CaImAn mmap filename issue
file_name_without_ext = os.path.basename(data_path).replace('.ome.tif', '')
mmap_path = os.path.join(tmp_dir, {file_name_without_ext}.pop() + '.mmap')
print(f"Saving mmap to: {mmap_path}")

# Convert to CaImAn movie
caiman_movie = cm.movie(data)

# Save as mmap
try:
    print(f"💾 Saving mmap to: {mmap_path}")
    caiman_movie.save(mmap_path, order='C')  # No memmap_dtype!
    print("✅ mmap save attempt complete")
except Exception as e:
    print(f"❌ Error saving mmap: {e}")
    exit()

if data.shape[0] == 0:
    print("❌ ERROR: Data is empty, cannot save mmap!")
    exit()

files = os.listdir(tmp_dir)
print("📂 Files in mmap directory:", files)
print("Files in temp dir:", os.listdir(tmp_dir))

# Get the actual .mmap path
mmap_file = None
for file in files:
    if file.startswith(file_name_without_ext) and file.endswith('.mmap'):
        mmap_file = os.path.join(tmp_dir, file)
        break

if not os.path.exists(mmap_file):
    print(f"❌ ERROR: mmap file does not exist! {mmap_file}")
    exit()

print(f"File size of mmap: {os.path.getsize(mmap_file)} bytes")

mmap_movie = cm.load(mmap_file)
print(mmap_movie.shape)

if mmap_movie.shape[0] == 0:
    print("❌ ERROR: mmap movie has zero frames!")
    exit()
else:
    print(f"✅ Loaded mmap movie with shape: {mmap_movie.shape}")

# Proceed with motion correction
print("🔄 Starting motion correction...")
mc = cm.motion_correction.MotionCorrect(data_path, 
                                        max_shifts=(6, 6), 
                                        strides=(48, 48), 
                                        overlaps=(24, 24))

if mc.min_mov is None:
    print("⚠️ WARNING: min_mov is None, setting it to 0")

mc.min_mov = 0  # Fix the NoneType issue

mc.motion_correct_rigid(save_movie=True)

# Reload corrected movie
print("🔄 Reloading corrected movie...")
motion_corrected_data = cm.load(mmap_file)
print(f"✅ Motion correction completed. Shape: {motion_corrected_data.shape}")


# Preprocessing and Deconvolution
print("🔄 Starting Preprocessing and Deconvolution...")
#mmap_file = np.array(mc.fname_tot_rig)
# Ensure the mmap file path is correct

# Prepare parameters for CNMF
params_dict = {
    # General parameters
    'fr': 30,  # Frame rate (Hz)
    'decay_time': 0.4,  # Typical decay time for calcium indicator
    'gSig': (3, 3),  # Gaussian width of neurons (in pixels)
    'gSiz': (13, 13),  # Spatial window size (~4*gSig)
    
    # Initialization parameters (merge into correct sections)
    'init_method': 'corr_pnr',  # Use correlation + peak-to-noise ratio for initialization
    'min_corr': 0.8,  # Minimum correlation for source extraction
    'min_pnr': 10,  # Minimum peak-to-noise ratio
    'ring_size_factor': 1.5,  # Ring size for initialization
    
    # Spatial and temporal processing
    'merge_thr': 0.85,  # Threshold for merging components
    'p': 1,  # Order of AR model for deconvolution
    'rf': None,  # Spatial extent of patches (None means whole FOV)
    
    # Computational settings
    'n_processes': 1,  # Number of processes to use
    'only_init': False,  # Perform full CNMF after initialization
}

# Properly create CNMFParams
opts = cnmf_params.CNMFParams()
opts.set(params_dict, val_dict = True)
opts.set('fnames', mmap_file)  # Set the filename for CNMF

# Initialize CNMF
print("🔄 Initializing CNMF...")
cnm = CNMF(n_processes=1, params=opts)

# Diagnostic function for memory mapping
def diagnose_memmap(movie):
    try:
        # Try memory mapping with verbose output
        mmap_file = cm.save_memmap(
            movie, 
            base_name='diagnostics', 
            dview=None, 
            order='C',
)
        
        print("Memory mapping successful!")
        print("Mmap file type:", type(mmap_file))
        print("Mmap file shape:", mmap_file.shape)
        
        # Try loading the memory-mapped file
        loaded_mmap = cm.load_memmap(mmap_file, mode='r')
        print("Loaded mmap type:", type(loaded_mmap))
        print("Loaded mmap shape:", loaded_mmap.shape)
        
        return mmap_file, loaded_mmap
    
    except Exception as e:
        print("Error during memory mapping:")
        print(str(e))
        return None, None

# Assuming 'motion_corrected' is your motion-corrected data
# Replace this with your actual motion-corrected data variable
mmap_diagnostic, loaded_diagnostic = diagnose_memmap(motion_corrected_data)

base_filename = 'histed_2101_test2_n2_200msSTIM_MMStack'

# Find the generated .mmap file
def find_recent_mmap(directory, base_name):
    mmap_files = [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.startswith(base_name) and f.endswith('.mmap')
    ]
    
    if not mmap_files:
        raise FileNotFoundError(f"No .mmap files found for {base_name}")
    
    # Return the most recently created file
    return max(mmap_files, key=os.path.getctime)

# Get the correct mmap file path
mmap_file = find_recent_mmap(tmp_dir, base_filename)
print(f"Using mmap file: {mmap_file}")

# Load the movie for visualization
loaded_movie = cm.load(mmap_file)

# Fit CNMF to the motion-corrected movie
print("🔄 Fitting CNMF to motion-corrected movie...")
cnm.fit(loaded_movie)

# Modify deconvolution parameters explicitly
d_params = {
    'method_deconvolution': 'oasis',  # Specify method
    'p': 1,  # AR model order
    'noise_range': [0.25, 0.5],  # Explicitly define as list of float values
    'noise_method': 'mean'
}

# Ensure clean numeric input
def prepare_deconvolution_input(temporal_components):
    # Convert to float64 to ensure compatibility
    temporal_components = np.array(temporal_components, dtype=np.float64)
    
    # Check for NaN or infinite values
    if np.isnan(temporal_components).any() or np.isinf(temporal_components).any():
        print("Warning: NaN or infinite values detected. Cleaning data.")
        temporal_components = np.nan_to_num(temporal_components)
    
    return temporal_components

# Modify deconvolution call
try:
    # Prepare temporal components
    cleaned_C = prepare_deconvolution_input(cnm.estimates.C)
    
    # Update estimates with cleaned components
    cnm.estimates.C = cleaned_C
    
    # Perform deconvolution with explicit parameters
    cnm.estimates.deconvolve(d_params, dff_flag=True)

except Exception as e:
    print(f"Deconvolution error: {e}")
    # Optional: print more diagnostic information
    print("Temporal components shape:", cnm.estimates.C.shape)
    print("Temporal components dtype:", cnm.estimates.C.dtype)

# Select components
print("🔍 Selecting ROIs...")

components = f'F:/histed/Coordinates/histed_2101_test2_n2_200msSTIM_MMStack.ome-roi_coords.csv'
components_df = pd.read_csv(components)
print(f"Selected components: {components_df}")
print(type(components_df))

cnm.estimates.select_components(idx_components=components_df, use_object=True)

# Visualize Results
plt.figure(figsize=(15, 5))

# Plot Spatial Components
plt.subplot(131)
cnm.estimates.plot_contours(img=loaded_movie.mean(axis=0))
plt.title('Extracted ROIs')

# Plot Temporal Components (Deconvolved Traces)
plt.subplot(132)
cnm.estimates.plot_contours(loaded_movie)
plt.title('Deconvolved Neural Activity')

# Plot Reconstructed Movie
plt.subplot(133)
reconstructed_movie = cnm.estimates.reconstruct_movie()
plt.imshow(reconstructed_movie.mean(axis=0), cmap='viridis')
plt.title('Reconstructed Movie')

plt.tight_layout()
plt.show()

# Save ROI data and traces
print("💾 Saving ROI data...")
np.savez('roi_analysis_results.npz', 
         spatial_components=cnm.estimates.A.toarray(), 
         temporal_components=cnm.estimates.C,
         deconvolved_traces=cnm.estimates.S)

# Optional: Save reconstructed movie
reconstructed_movie.save('reconstructed_movie.tiff')

print("✅ Preprocessing, Deconvolution, and ROI Analysis Complete!")

# Fluorescence Analysis
print("\n📊 Fluorescence Analysis Results:")
for i, trace in enumerate(cnm.estimates.C):
    print(f"ROI {i+1}:")
    print(f"  Peak Activity: {np.max(trace)}")
    print(f"  Mean Fluorescence: {np.mean(trace)}")
    print(f"  Activity Duration: {np.sum(trace > 0)} frames")


gsig_tmp = (3,3)
correlation_image, peak_to_noise_ratio = cm.summary_images.correlation_pnr(images[::max(T//1000, 1)], # subsample if needed
                                                                           gSig=gsig_tmp[0], # used for filter
                                                                           swap_dim=False)