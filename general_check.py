import sys
import os
import subprocess

def check_python_version():
    print("Python version:", sys.version)

def check_pip_version():
    try:
        import pip
        print("Pip version:", pip.__version__)
    except ImportError:
        print("Pip is not installed.")

def check_installed_packages():
    try:
        import pip
        installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'list'])
        print("Installed packages:")
        print(installed_packages.decode())
    except Exception as e:
        print("Error checking installed packages:", e)

def check_skimage_installation():
    try:
        import skimage
        print("scikit-image version:", skimage.__version__)
    except ImportError:
        print("scikit-image is not installed.")

def main():
    check_python_version()
    check_pip_version()
    check_installed_packages()
    check_skimage_installation()

if __name__ == "__main__":
    main()


import tifffile
import numpy as np

with tifffile.TiffFile('C:/Users/sofik/Downloads/rgi48_testcode.ome.tif') as tif:
    data = tif.asarray()

print(f"Loaded data shape: {data.shape}")
print(f"Number of non-zero frames: {np.sum(np.any(data != 0, axis=(1,2)))}")
print(f"Number of zero frames: {np.sum(np.all(data == 0, axis=(1,2)))}")

non_zero_frames = np.any(data != 0, axis=(1,2))
valid_data = data[non_zero_frames]
print(f"Shape of valid data: {valid_data.shape}")
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(data[0], cmap='gray')
plt.title("First Frame")
plt.subplot(122)
plt.imshow(data[-1], cmap='gray')
plt.title("Last Frame")
plt.show()

def process_tiff(file_path):
    with tifffile.TiffFile(file_path) as tif:
        data = tif.asarray()
    
    non_zero_frames = np.any(data != 0, axis=(1,2))
    valid_data = data[non_zero_frames]
    
    if valid_data.shape[0] < data.shape[0]:
        print(f"Warning: {data.shape[0] - valid_data.shape[0]} frames were zero-filled")
    
    return valid_data

# Use this function in your analysis
processed_data = process_tiff('C:/Users/sofik/Downloads/rgi48_testcode.ome.tif')

with tifffile.TiffFile('C:/Users/sofik/Downloads/rgi48_testcode.ome.tif') as tif:
    ome_metadata = tif.ome_metadata
print(ome_metadata)