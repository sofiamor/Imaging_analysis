import tifffile as tiff
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.restoration import rolling_ball
from bm3d import bm3d, BM3DProfile



# Path to input TIFF file
input_tiff_path = 'F:/histed/histed_21.1_test2_n2_200msSTIM/histed_21.1_test2_n2_200msSTIM/histed_21.1_test2_n2_200msSTIM_MMStack.ome.tif'
base_name = os.path.basename(input_tiff_path)
file_name_without_ext = os.path.splitext(base_name)[0]
output_tiff_path =  f'F:/histed/df_over_f0/{file_name_without_ext}_preprocessingg.tif'

tiff_stack = tiff.imread(input_tiff_path)  # Load as a 3D array (frames, height, width)


# Preallocate for processed stack
processed_stack = []

# Parameters
rolling_ball_radius = 200  # Adjust based on your background gradient size
sigma = 0.1  # Noise standard deviation for BM3D

# Process each frame in the stack
for i, frame in enumerate(tiff_stack):
    print(f"Processing frame {i+1}/{tiff_stack.shape[0]}...")

    # Step 1: Apply sliding paraboloid (background subtraction)
    background = rolling_ball(frame, radius=rolling_ball_radius)
    corrected_frame = frame - background

    # Ensure no negative values after subtraction
    corrected_frame = np.clip(corrected_frame, 0, 255).astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Step 2: Apply BM3D denoising
    denoised_frame = bm3d(corrected_frame, sigma_psd=sigma)

    # Convert back to 8-bit format for saving
    denoised_frame = (denoised_frame * 255).astype(np.uint8)

    # Append the processed frame
    processed_stack.append(denoised_frame)

# Stack all processed frames back into a 3D array
processed_stack = np.array(processed_stack)

tiff.imwrite(output_tiff_path, processed_stack, photometric='minisblack')

print(f"Processed TIFF file saved to {output_tiff_path}")

