import tifffile
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.image import ImageSeries
from datetime import datetime

# 🔄 Step 1: Load the TIFF File (Assuming it's a T-Stack)
tif_path = 'F://histed//histed_2101_test2_n2_200msSTIM//histed_2101_test2_n2_200msSTIM//histed_2101_test2_n2_200msSTIM_MMStack.ome.tif'
tif_data = tifffile.imread(tif_path)  # Shape should be (T, X, Y) or (T, X, Y, C)

# 🔄 Step 2: Define Imaging Parameters
frame_rate = 30.0  # Adjust to your actual frame rate (Hz)

# 🔄 Step 3: Create NWB File
nwbfile = NWBFile(
    session_description="T-Stack TIFF to NWB conversion",
    identifier="TStack_Example",
    session_start_time=datetime.now()
)

# 🔄 Step 4: Add TIFF T-Stack Data as an ImageSeries
image_series = ImageSeries(
    name="TStack_Image",
    data=tif_data,  # NumPy array with shape (T, X, Y)
    unit="pixels",
    format="tiff",
    rate=frame_rate  # Frame rate in Hz
)
nwbfile.add_acquisition(image_series)

# 🔄 Step 5: Save NWB File
nwb_path = "F:/histed/TESThisted2101_200ms_tstack.nwb"
with NWBHDF5IO(nwb_path, "w") as io:
    io.write(nwbfile)

print(f"✅ NWB file saved: {nwb_path}")
