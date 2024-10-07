import tifffile as tiff
import numpy as np
import pandas as pd
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector
import os
from matplotlib.patches import Ellipse

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
    fig, ax = plt.subplots(figsize=(10, 10))
    ellipse = Ellipse((center_x, center_y), width, height, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(ellipse)
    circles.append(ellipse)
    plt.draw()

def onclick(event):
    """Handle deletion of ROIs."""
    if event.button == 2:  # Right-click deletes the nearest ellipse
        for i, ellipse in reversed(list(enumerate(circles))):
            if ellipse.contains_point((event.xdata, event.ydata)):
                ellipse.remove()  # Remove the ellipse from the plot
                circles.pop(i)  # Remove the ellipse from the list
                rois.pop(i)  # Remove the corresponding ROI data
                plt.draw()
                break


def select_rois(img_rescale, onlick):
    """Display image and handle ROI selection."""
    global rois, circles
    rois = []
    circles = []

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_rescale, cmap='gray')

    # Connect the EllipseSelector to the onselect function
    toggle_selector = EllipseSelector(ax, onselect, useblit=True,
                                      button=[1], minspanx=5, minspany=5,
                                      spancoords='pixels', interactive=True)

    # Connect right-click event to delete ellipses
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.title("Select ROIs by dragging. Right-click to delete. Close when done.")
    plt.show(block=True)

    def onclick(event):
    #"""Handle deletion of ROIs."""
    
        if event.button == 2:  # Right-click deletes the nearest ellipse
            for i, ellipse in reversed(list(enumerate(circles))):
                if ellipse.contains_point((event.xdata, event.ydata)):
                    ellipse.remove()  # Remove the ellipse from the plot
                    circles.pop(i)  # Remove the ellipse from the list
                    rois.pop(i)  # Remove the corresponding ROI data
                    plt.draw()
                    break

        return rois

img_rescale = np.random.rand(100, 100)

# Call the ROI selection function with the rescaled image
rois = select_rois(img_rescale)
print("Final selected ROIs:", rois)