import tifffile as tiff
import numpy as np
import pandas as pd
from skimage import filters, measure, morphology, segmentation, feature, exposure
import matplotlib.pyplot as plt

# Load the TIFF file
tiff_path = 'C:/Users/sofik/Downloads/rgi48_testcode.ome.tif'
tiff_stack = tiff.imread(tiff_path)

print(f"Loaded TIFF stack with shape {tiff_stack.shape}")

# Use a maximum intensity projection
max_projection = np.mean(tiff_stack, axis=0)

# Enhance contrast
p2, p98 = np.percentile(max_projection, (13, 95))
output_range = (0, 240)
img_rescale = exposure.rescale_intensity(max_projection, in_range=(p2, p98), out_range=output_range)

# Apply Gaussian filter to reduce noise
img_smooth = filters.gaussian(img_rescale, sigma=1)

# Detect local peaks in the smoothed grayscale image
distance = 10  # Minimum distance between peaks
coords = feature.peak_local_max(img_smooth, min_distance=distance, threshold_abs=filters.threshold_otsu(img_smooth))

# Create a mask of peaks
mask = np.zeros(img_smooth.shape, dtype=bool)
mask[tuple(coords.T)] = True

# Label the peaks
markers = measure.label(mask)

# Apply watershed segmentation directly on the smoothed grayscale image
labels = segmentation.watershed(-img_smooth, markers, mask=img_smooth > filters.threshold_otsu(img_smooth))


# Calculate Otsu threshold
otsu_threshold = filters.threshold_otsu(img_smooth)

# Adjust the threshold with a scaling factor
scale_factor = 2.1  # Adjust this value to make the threshold more or less stringent
adjusted_threshold = otsu_threshold * scale_factor

# Create a binary image using the adjusted threshold
binary_image = img_smooth > adjusted_threshold

# Remove small objects
cleaned_binary_image = morphology.remove_small_objects(binary_image, min_size=100)

# Label connected regions
labeled_image = measure.label(cleaned_binary_image)

# Use region properties to filter out cells based on size and intensity
min_size = 100  # Minimum size of a cell in pixels
max_size = 50000  # Maximum size of a cell in pixels
min_intensity = np.percentile(img_smooth, 80)  # Minimum mean intensity of a cell

regions = measure.regionprops(labeled_image, intensity_image=img_smooth)
filtered_regions = [region for region in regions 
                    if min_size <= region.area <= max_size 
                    and region.mean_intensity > min_intensity]


# Filter regions based on size and intensity
min_size = 100  # Minimum size of a cell in pixels
max_size = 30000  # Maximum size of a cell in pixels
min_intensity = np.percentile(img_smooth, 90)  # Minimum mean intensity of a cell

filtered_regions = [region for region in regions 
                    if min_size <= region.area <= max_size 
                    and region.mean_intensity > min_intensity]

# Display the detected cells
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cleaned_binary_image, cmap='gray')
for region in filtered_regions:
    minr, minc, maxr, maxc = region.bbox
    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
ax.set_title('Detected Cells')
plt.show()

# Display the detected cells
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(max_projection, cmap='gray')
for region in filtered_regions:
    minr, minc, maxr, maxc = region.bbox
    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
ax.set_title('Detected Cells')
plt.show()

# Extract fluorescence data
fluorescence_data = []
for region in filtered_regions:
    minr, minc, maxr, maxc = region.bbox
    roi_intensities = []
    for frame in tiff_stack:
        roi = frame[minr:maxr, minc:maxc]
        roi_intensity = np.mean(roi)
        roi_intensities.append(roi_intensity)
    fluorescence_data.append(roi_intensities)

# Convert the data to a pandas DataFrame
fluorescence_df = pd.DataFrame(fluorescence_data).T
fluorescence_df.columns = [f'Cell_{i+1}' for i in range(len(filtered_regions))]

print(fluorescence_df.head())

# Save the DataFrame to a CSV file
output_path = 'C:/Users/sofik/Desktop/rgi48.csv'
fluorescence_df.to_csv(output_path, index=False)
print(f"Saved fluorescence data to {output_path}")
