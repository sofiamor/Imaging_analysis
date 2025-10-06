import tifffile as tiff
import numpy as np
import pandas as pd
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector
import os
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Ellipse
import PySimpleGUI as sg

# Load the TIFF file and process
#tiff_path = 'Z:/smorou/Imaging/Voltage_imaging/FORCE/force_day4/FORCE1F_DAY4_SLC2_VID3_75%_2x2/FORCE1F_DAY4_SLC2_VID3_75%_2x2_MMStack.ome.tif'
tiff_path = 'F:/IMAGING/ak3_15_vid3_stim200/ak3_15_vid3_stim200_MMStack.ome.tif'
base_name = os.path.basename(tiff_path)
file_name_without_ext = os.path.splitext(base_name)[0]
print(f"Processing file: {file_name_without_ext}")

# Parameters for chunked processing
chunk_size = 100  # Number of frames to process at a time
output_max_projection_path = f"{file_name_without_ext}_max_projection.npy"

# Initialize the max projection array
max_projection = None
frame_count = 0

try:
    with tiff.TiffFile(tiff_path) as tif:
        num_pages = len(tif.pages)
        page_shape = tif.pages[0].asarray().shape
        print(f"TIFF file has {num_pages} pages, each with shape {page_shape}")

        # Process the TIFF file in chunks
        for start in range(0, num_pages, chunk_size):
            end = min(start + chunk_size, num_pages)
            print(f"Processing frames {start} to {end - 1}")

            # Load frames in the current chunk
            chunk = []
            for i in range(start, end):
                page_data = tif.pages[i].asarray()
                if np.any(page_data != 0):  # Check for valid data
                    chunk.append(page_data.astype(np.float32))
                else:
                    print(f"Skipping frame {i} due to missing or incomplete data.")

            # Convert chunk to a NumPy array
            if chunk:
                chunk_array = np.stack(chunk, axis=0)  # Shape: (chunk_size, H, W)
                chunk_array[chunk_array == 0] = np.nan  # Replace zeros with NaN

                # Compute the cumulative max projection
                if max_projection is None:
                    max_projection = np.nanmean(chunk_array, axis=0)
                else:
                    max_projection = np.nanmean(np.dstack([max_projection, np.nanmean(chunk_array, axis=0)]), axis=2)

                frame_count += chunk_array.shape[0]
    

    print(f"Processed {frame_count} valid frames out of {num_pages}.")
    print(f"Max projection shape: {max_projection.shape}")

    # Save the max projection
    np.save(output_max_projection_path, max_projection)
    print(f"Max projection saved to: {output_max_projection_path}")

except MemoryError as e:
    print("MemoryError encountered. Consider reducing the chunk size further.")
    raise e

# Save the max projection if needed
output_path = f"{file_name_without_ext}_max_projection.npy"
np.save(output_path, max_projection)
print(f"Max projection saved to: {output_path}")

# Enhance contrast
from skimage import exposure
p2, p98 = np.percentile(max_projection, (0, 100))
output_range = (0, 240)
img_rescale = exposure.rescale_intensity(max_projection, in_range=(p2, p98), out_range=output_range)

# Initialize variables for ROIs
rois = []
circles = []

# PySimpleGUI layout
layout = [
    [sg.Text('Select ROIs on the image. Close the window when done.')],
    [sg.Canvas(key='-CANVAS-')],
    [sg.Button('Undo Last ROI'), sg.Button('Save and Exit')]
]

# Create the PySimpleGUI window
window = sg.Window('ROI Selector', layout, finalize=True)

# Embed the Matplotlib figure into the PySimpleGUI window
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_rescale, cmap='gray')
canvas = FigureCanvasTkAgg(fig, window['-CANVAS-'].TKCanvas)
canvas.draw()
canvas.get_tk_widget().pack()

# Function to handle drawing and selecting ROIs
def onselect(event_click, event_release):
    x1, y1 = event_click.xdata, event_click.ydata
    x2, y2 = event_release.xdata, event_release.ydata

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    # Store the ROI
    rois.append((center_y, center_x, width, height))

    # Draw the ellipse on the plot
    ellipse = Ellipse((center_x, center_y), width, height, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(ellipse)
    circles.append(ellipse)
    plt.draw()
    canvas.draw()

# Function to undo the last ROI
def undo_last_roi():
    if circles:
        circle = circles.pop()
        circle.remove()
        rois.pop()
        plt.draw()
        canvas.draw()

# Connect the Matplotlib EllipseSelector
from matplotlib.widgets import EllipseSelector
toggle_selector = EllipseSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5,
                                  spancoords='pixels', interactive=True,
                                  props=dict(facecolor='none', edgecolor='red'))

# Event loop
while True:
    event, _ = window.read()
    if event in (sg.WIN_CLOSED, 'Save and Exit'):
        break
    elif event == 'Undo Last ROI':
        undo_last_roi()

window.close()

# Extract fluorescence data for each ROI
fluorescence_data = []
coordinates = []

with tiff.TiffFile(tiff_path) as tif:
    num_pages = len(tif.pages)  # Get the total number of pages (frames)

    for (center_y, center_x, width, height) in rois:
        roi_intensities = []  # Store fluorescence data for this ROI
        for i in range(num_pages):  # Loop over all frames
            frame = tif.pages[i].asarray()  # Read the frame
            Y, X = np.ogrid[:frame.shape[0], :frame.shape[1]]
            mask = ((X - center_x) ** 2 / (width / 2) ** 2 + (Y - center_y) ** 2 / (height / 2) ** 2) <= 1
            roi = frame[mask]
            roi_intensity = np.mean(roi) if roi.size > 0 else np.nan
            roi_intensities.append(roi_intensity)

        fluorescence_data.append(roi_intensities)
        coordinates.append((center_x, center_y))

print(len(fluorescence_data))

# Convert the data to a DataFrame
fluorescence_df = pd.DataFrame(fluorescence_data).T
print(fluorescence_df.shape)
print(len(fluorescence_df))
fluorescence_df.columns = [f'Cell_{i+1}' for i in range(len(rois))]
fluorescence_df = fluorescence_df.dropna(how='all').dropna(axis=1, how='all')
fluorescence_df = fluorescence_df.dropna(axis=1, thresh=int(0.9 * len(fluorescence_df)))

# Save the data
output_path1 = f'F:/histed/Traces/{file_name_without_ext}_traces.csv'
output_path2 = f'F:/histed/Coordinates/{file_name_without_ext}_coords.csv'

with open(output_path1, 'w') as f:
    f.write(fluorescence_df.to_csv())
with open(output_path2, 'w') as f:
    f.write(pd.DataFrame(coordinates, columns=['X', 'Y']).to_csv())

#fluorescence_df.to_csv(output_path1, index=False)
#pd.DataFrame(coordinates, columns=['X', 'Y']).to_csv(output_path2, index=False)

print(f"Saved fluorescence data to {output_path1}")
print(f"Saved coordinates to {output_path2}")
