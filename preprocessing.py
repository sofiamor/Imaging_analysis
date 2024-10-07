import tifffile as tiff
import numpy as np
import pandas as pd
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector
import os
from matplotlib.patches import Ellipse

# Load the TIFF file
tiff_path = 'Z:/smorou/Imaging/Calcium_Imaging/FV3/4X4BIN_100%_GCAMP_50%_CHRIMSON1/4X4BIN_100%_GCAMP_50%_CHRIMSON1_MMStack.ome.tif'

# Extract the base name (file name without path) and remove the extension
base_name = os.path.basename(tiff_path)
file_name_without_ext = os.path.splitext(base_name)[0]
print(f"Processing file: {file_name_without_ext}")

# Open the TIFF file
with tiff.TiffFile(tiff_path) as tif:
    num_pages = len(tif.pages)  # The maximum number of pages based on metadata
    page_shape = tif.pages[0].asarray().shape
    print(f"TIFF file has up to {num_pages} pages, each with shape {page_shape}")

    # Initialize an empty list to hold valid frames
    valid_frames = []

    # Read the TIFF pages and only keep valid frames
    for i in range(num_pages):

        page_data = tif.pages[i].asarray()
        
        # Check if the frame is valid (not all zeros)
        if np.any(page_data != 0):  # Ensure at least one pixel is non-zero
            valid_frames.append(page_data.astype(np.float32))
        else:
            print(f"Stopping at frame {i} due to missing or incomplete data.")
            break
    
    # Convert the list of valid frames to a NumPy array
    tiff_stack = np.array(valid_frames)

    # Check the shape of the valid tiff_stack
    print(f"tiff_stack shape: {tiff_stack.shape}")
    if tiff_stack.size == 0:
        print("No valid frames found!")
    else:
        print(f"Valid frames loaded: {tiff_stack.shape[0]} frames out of {num_pages}")

    # Calculate maximum projection, ignoring NaNs (converted from zeros)
    tiff_stack[tiff_stack == 0] = np.nan
    max_projection = np.nanmean(tiff_stack[45:], axis=0)

    # Proceed with further steps like saving the max projection
    print(f"Max projection calculated successfully with shape: {max_projection.shape}")
    # You can now save the max projection or process it further

# Use a maximum intensity projection
max_projection = np.mean(tiff_stack, axis=0)

# Enhance contrast
p2, p98 = np.percentile(max_projection, (0, 100))
output_range = (0, 240)
img_rescale = exposure.rescale_intensity(max_projection, in_range=(p2, p98), out_range=output_range)

rois = []
circles = []

def onselect(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    
    # Store the ROI as (center_y, center_x, width, height)
    rois.append((center_y, center_x, width, height))
    
    # Draw the ellipse on the plot
    ellipse = Ellipse((center_x, center_y), width, height, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(ellipse)
    circles.append(ellipse)
    plt.draw()

def circle_constraint(box_extents):
    center_x, center_y = (box_extents[0] + box_extents[1]) / 2, (box_extents[2] + box_extents[3]) / 2
    radius = max(abs(box_extents[1] - box_extents[0]), abs(box_extents[3] - box_extents[2])) / 2
    return (center_x - radius, center_x + radius, center_y - radius, center_y + radius)

def onclick(event):
    if event.button == 3:  # Right mouse button
        for i, circle in enumerate(circles):
            if circle.contains_point((event.xdata, event.ydata)):
                circle.remove()
                circles.pop(i)
                rois.pop(i)
                plt.draw()
                break

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_rescale, cmap='gray')

toggle_selector = EllipseSelector(ax, onselect, useblit=True,
                                  button=[1], minspanx=5, minspany=5,
                                  spancoords='pixels', interactive=True,
                                  props=dict(facecolor='none', edgecolor='red'))

fig.canvas.mpl_connect('button_press_event', onclick)

plt.title("Select circular ROIs by dragging. Right-click to erase. Close the window when done.")
plt.show()
print(f"Selected {len(rois)} ROIs.")

# Extract fluorescence data and the respective coordinates for the selected ROIs
fluorescence_data = []
coordinates = []

for (center_y, center_x, width, height) in rois:
    roi_intensities = []
    for frame in tiff_stack:
        # Create an elliptical mask
        Y, X = np.ogrid[:frame.shape[0], :frame.shape[1]]
        mask = ((X - center_x) ** 2 / (width / 2) ** 2 + (Y - center_y) ** 2 / (height / 2) ** 2) <= 1
        
        # Apply the mask to extract the ROI
        roi = frame[mask]
        roi_intensity = np.mean(roi) if roi.size > 0 else np.nan  # Use np.nan for missing data
        roi_intensities.append(roi_intensity)

    # Append the fluorescence data
    fluorescence_data.append(roi_intensities)
    
    # Append the coordinates as a tuple (center_x, center_y)
    coordinates.append((center_x, center_y))

# Convert the data to a pandas DataFrame
fluorescence_df = pd.DataFrame(fluorescence_data).T
fluorescence_df.columns = [f'Cell_{i+1}' for i in range(len(rois))]

# Drop rows with all NaN values
fluorescence_df = fluorescence_df.dropna(how='all')

# Drop columns with all NaN values
fluorescence_df = fluorescence_df.dropna(axis=1, how='all')

# Drop columns where NaN values are present, but only in the case of a significant amount of missing data
# Uncomment the following line if you want to drop columns with a high percentage of missing data
fluorescence_df = fluorescence_df.dropna(axis=1, thresh=int(0.9 * len(fluorescence_df)))

# Add your additional text
text1 = '_traces'
text2 = '_coords'

# Save the DataFrame to a CSV file
output_path1 = f'Z:/smorou/Analysis/imaging_analysis/CalciumIm/Traces/{file_name_without_ext}{text1}.csv'
output_path2 = f'Z:/smorou/Analysis/imaging_analysis/CalciumIm/Coordinates/{file_name_without_ext}{text2}.csv'

fluorescence_df.to_csv(output_path1, index=False)
pd.DataFrame(coordinates, columns=['X', 'Y']).to_csv(output_path2, index=False)

print(f"Saved fluorescence data to {output_path1}")
print(f"Saved coordinates to {output_path2}")
