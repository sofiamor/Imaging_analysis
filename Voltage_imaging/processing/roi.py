import numpy as np
import os
import matplotlib.pyplot as plt
import json
import copy
import scipy.ndimage
import skimage.filters
from skimage import morphology, measure
from read_roi import read_roi_file

def find_roi_source_folders(base_day_folder):
    roi_candidates = {}

    for cell_folder in [os.path.join(base_day_folder, d) for d in os.listdir(base_day_folder) if os.path.isdir(os.path.join(base_day_folder, d))]:
        protocol_folders = [os.path.join(cell_folder, d) for d in os.listdir(cell_folder)
                            if os.path.isdir(os.path.join(cell_folder, d)) and "population" not in d.lower()]

        # Try to find SD or sensitivity folder
        sd_folder = next((p for p in protocol_folders if "spike" in os.path.basename(p).lower()), None)
        sens_folder = next((p for p in protocol_folders if "sens" in os.path.basename(p).lower()), None)

        roi_folder = sd_folder if sd_folder else sens_folder
        if roi_folder:
            roi_candidates[cell_folder] = roi_folder
            
    save_path = os.path.join(base_day_folder, "roi_sources.json")

    return roi_candidates

def load_roi_sources_json(base_dir):
    save_path = os.path.join(base_dir, "roi_sources.json")
    with open(save_path, "w") as f:
        json.dump(roi_sources, f, indent=2)
    print(f"✅ Saved all ROI sources in {save_path}")
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            return json.load(f)
    else:
        print("⚠️ No roi_sources.json found.")
        return {}

def calculate_spatial_average(data, dim_spatial_average_kernel_pix=3, disp_fig=False):
    data[np.isnan(data)] = np.nanmean(data)
    
    spatial_average_kernel = np.ones([dim_spatial_average_kernel_pix, dim_spatial_average_kernel_pix])
    cent_pix = np.ceil(np.divide(dim_spatial_average_kernel_pix - 1, 2)).astype(int)
    spatial_average_kernel[cent_pix, cent_pix] = 0

    if len(data.shape) > 2:
        spatial_average = np.mean([scipy.ndimage.convolve(_, spatial_average_kernel, origin=0) for _ in data], axis=0)
        norm = np.mean([scipy.ndimage.convolve(_, spatial_average_kernel, origin=0) for _ in np.ones_like(data)], axis=0)
        
    else:
        spatial_average = scipy.ndimage.convolve(data, spatial_average_kernel, origin=0)
        norm = scipy.ndimage.convolve(np.ones_like(data), spatial_average_kernel, origin=0)

    spatial_average = spatial_average/norm

    if disp_fig:
        fig, ax = plt.subplots()
        ax.imshow(spatial_average)
        plt.show()

    return spatial_average

def generate_segmentation_masks(img_to_segment, approx_radius_scanless_excitation_spot, disp_img=False):

    threshold_value = skimage.filters.threshold_otsu(img_to_segment)
    init_segmentation = np.ones_like(img_to_segment)
    init_segmentation[img_to_segment < threshold_value] = 0

    # keep largest segment
    labels = skimage.measure.label(init_segmentation, return_num=False)
    init_segmentation = labels == np.argmax(np.bincount(labels.flat, weights=init_segmentation.flat))
    
    tmp = skimage.morphology.disk(radius=approx_radius_scanless_excitation_spot/2)

    dilated_segmentation = skimage.morphology.dilation(init_segmentation, tmp)

    anti_cell_mask = copy.deepcopy(dilated_segmentation)
    anti_cell_mask = scipy.ndimage.binary_fill_holes(anti_cell_mask).astype(np.float64)
    anti_cell_mask = np.abs(anti_cell_mask - 1).astype(np.float64)

    if np.sum(anti_cell_mask) == 0: 
        anti_cell_mask[0,0] = 1
        anti_cell_mask[0,-1] = 1
        anti_cell_mask[-1,0] = 1
        anti_cell_mask[-1,-1] = 1

    if disp_img is True:      
        fig, axs = plt.subplots(nrows=1, ncols=4)
        # fig.set_size_inches(12,4)
        ax0, ax1, ax2, ax3 = axs
        ax0.imshow(img_to_segment)
        ax0.set_title("To segment")
        ax1.imshow(init_segmentation*img_to_segment)
        ax1.set_title("Initial segmentation")
        ax2.imshow(dilated_segmentation*img_to_segment)
        ax2.set_title("Dilated segmentation")
        ax3.imshow(anti_cell_mask*img_to_segment)
        ax3.set_title("Anti-cell mask")
        [ax.axis("off") for ax in axs]
        # fig.suptitle("Segmentation")
        plt.tight_layout()
        plt.show()

    return init_segmentation, dilated_segmentation, anti_cell_mask

def create_masks_from_sd_recording(sd_tiff, approx_radius_scanless_excitation_spot=15,
                                   spatial_kernel_pix=15, dff_percentile=90,
                                   disp_img=True):
    """
    Generate segmentation masks from SD70-10-1s recording:
    - Computes spatially averaged image
    - Computes ΔF/F activity map
    - Combines both for robust segmentation
    - Returns ROI (cell) and anti-ROI (background) masks
    """

    # Step 1. Replace NaNs
    data = sd_tiff.astype(float)
    data[np.isnan(data)] = np.nanmean(data)
        # Step 2. Compute spatial average (smoothing)
    spatial_average = calculate_spatial_average(data, dim_spatial_average_kernel_pix=spatial_kernel_pix, disp_fig=False)

    # Step 3. Compute ΔF/F per pixel
    f0 = np.percentile(data, 10, axis=0)  # baseline (10th percentile)
    fmax = np.percentile(data, dff_percentile, axis=0)
    dff = (fmax - f0) / np.maximum(f0, 1e-6)

    # Step 4. Normalize maps and combine them
    spatial_norm = (spatial_average - np.min(spatial_average)) / (np.max(spatial_average) - np.min(spatial_average))
    dff_norm = (dff - np.min(dff)) / (np.max(dff) - np.min(dff))
    combined_img = 0.5 * spatial_norm + 0.5 * dff_norm

    # Step 5. Use your segmentation function on the combined image
    init_segmentation, dilated_segmentation, anti_cell_mask = generate_segmentation_masks(
        combined_img,
        approx_radius_scanless_excitation_spot=approx_radius_scanless_excitation_spot,
        disp_img=disp_img
    )

    return init_segmentation, dilated_segmentation, anti_cell_mask, combined_img

def load_roi_mask(roi_path, shape):
    roi_dict = read_roi_file(roi_path)
    mask = np.zeros(shape, dtype=bool)

    for name, roi in roi_dict.items():
        if 'x' in roi and 'y' in roi:
            # Polygon or freehand ROI
            rr, cc = polygon(roi['y'], roi['x'], shape)
            mask[rr, cc] = True

        elif all(k in roi for k in ('left', 'top', 'width', 'height')):
            # Rectangle or oval ROI
            x0, y0 = roi['left'], roi['top']
            x1, y1 = x0 + roi['width'], y0 + roi['height']
            mask[y0:y1, x0:x1] = True

        else:
            print(f"⚠️ Unknown ROI type for '{name}', keys: {list(roi.keys())}")
    return mask