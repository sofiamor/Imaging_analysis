import tifffile as tiff
import numpy as np
import pandas as pd
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join 
import os
from matplotlib.patches import Ellipse


# Load the TIFF file
tiff_path = 'C:/Users/sofik/Downloads/FV4/4X4BIN_100_ GCAMP_50_ CHRIMSON 2 5MIN/4X4BIN_100_ GCAMP_50_ CHRIMSON 2 5MIN_MMStack.ome.tif'

# Extract the base name (file name without path) and remove the extension
base_name = os.path.basename(tiff_path)
file_name_without_ext = os.path.splitext(base_name)[0]
print(f"Processing file: {file_name_without_ext}")

tiff_stack = tiff.imread(tiff_path)

print(f"Loaded TIFF stack with shape {tiff_stack.shape}")

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


# Extract fluorescence data and the respective coordinates for the selected ROIs
fluorescence_data = []
coordinates = []

for (center_y, center_x, width, height) in rois:
    roi_intensities = []
    for frame in tiff_stack:
        # Create a circular mask
        Y, X = np.ogrid[:frame.shape[0], :frame.shape[1]]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = ((X - center_x) ** 2 / (width / 2) ** 2 + (Y - center_y) ** 2 / (height / 2) ** 2) <= 1
        
        # Apply the mask to extract the ROI
        roi = frame[mask]
        roi_intensity = np.mean(roi)
        roi_intensities.append(roi_intensity)

    # Append the fluorescence data
    fluorescence_data.append(roi_intensities)
    
    # Append the coordinates as a tuple (center_x, center_y)
    coordinates.append((center_x, center_y))


# Now `coordinates` will contain the list of (roi_x, roi_y) pairs
print(coordinates)

# Convert the data to a pandas DataFrame
fluorescence_df = pd.DataFrame(fluorescence_data).T
fluorescence_df.columns = [f'Cell_{i+1}' for i in range(len(rois))]

print(fluorescence_df.head())
# Add your additional text
text1 = '_traces'
text2 = '_coords'

# Save the DataFrame to a CSV file
output_path1 = f'C:/Users/sofik/Desktop/Traces/{file_name_without_ext}{text1}.csv'
output_path2 = f'C:/Users/sofik/Desktop/Coordinates/{file_name_without_ext}{text2}.csv'

fluorescence_df.to_csv(output_path1, index=False)
pd.DataFrame(coordinates, columns=['X', 'Y']).to_csv(output_path2, index=False)

print(f"Saved fluorescence data to {output_path1}")
print(f"Saved coordinates to {output_path2}")