import numpy as np
import os
os.environ['MPLBACKEND'] = 'Qt5Agg'  # Set backend before importing matplotlib
import matplotlib
matplotlib.use('Qt5Agg')  # Explicitly set backend
import matplotlib.pyplot as plt

def select_rois(stack, method="ask", roi_radius=5, predefined_coords=None):
    """
    Select ROIs from an image stack.

    Parameters
    ----------
    stack : np.ndarray
        Image stack (frames, height, width)
    method : str
        "ask" = ask user whether manual or auto
        "manual" = interactive ROI selection
        "auto" = use predefined coordinates
    roi_radius : int
        Radius of circular ROI (pixels)
    predefined_coords : list of tuples
        [(x1, y1), (x2, y2), ...] for automatic ROIs

    Returns
    -------
    rois : list of dicts
        Each dict: {"x": x, "y": y, "radius": r, "mask": np.ndarray}
    """
    # Ask user if needed
    if method == "ask":
        choice = input("Select ROI method (manual/auto): ").lower()
    else:
        choice = method

    rois = []

    if choice == "manual":
        rois = _manual_rois(stack, roi_radius)
    elif choice == "auto":
        if predefined_coords is None:
            raise ValueError("For automatic ROI, you must provide coordinates!")
        rois = _auto_rois(stack, predefined_coords, roi_radius)
    else:
        raise ValueError("Unknown ROI selection method.")

    return rois


def _manual_rois(stack, roi_radius):
    """
    Interactive ROI selection using matplotlib clicks.
    """
    mean_frame = np.mean(stack, axis=0)
    rois = []

    fig, ax = plt.subplots()
    ax.imshow(mean_frame, cmap="gray")
    ax.set_title("Click to select ROIs. Close the window when done.")

    coords = plt.ginput(n=-1, timeout=0)  # n=-1 = unlimited clicks
    plt.close(fig)

    for x, y in coords:
        mask = _make_circular_mask(stack.shape[1], stack.shape[2], center=(x, y), radius=roi_radius)
        rois.append({"x": x, "y": y, "radius": roi_radius, "mask": mask})

    print(f"Selected {len(rois)} ROIs.")
    return rois


def _auto_rois(stack, coords_list, roi_radius):
    """
    Automatic ROIs from given coordinates.
    """
    rois = []
    for x, y in coords_list:
        mask = _make_circular_mask(stack.shape[1], stack.shape[2], center=(x, y), radius=roi_radius)
        rois.append({"x": x, "y": y, "radius": roi_radius, "mask": mask})
    return rois


def _make_circular_mask(h, w, center=None, radius=None):
    """
    Create a boolean mask with a circle.
    """
    Y, X = np.ogrid[:h, :w]
    if center is None:
        center = (w//2, h//2)
    if radius is None:
        radius = min(h, w)//4

    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask
