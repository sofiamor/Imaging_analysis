import numpy as np
import os
import tifffile


def load_all_tiffs_from_day(base_day_folder, visualize_unexpected=True):
    """
    Loads all TIFF files recursively from a day folder.
    Expected structure:
        day_folder/cell_number/protocol_type/file_folder/file.tif
    Returns a dictionary organized as:
        data_dict[cell][protocol][trial_name] = numpy array
    """
    data_dict = {}
    sd70_path = None
    unexpected_files = []
    
    for root, dirs, files in os.walk(base_day_folder):
        # --- Skip folders containing 'population' in their path ---
        if "population" in root.lower():
            print(f"⏭️  Skipping folder (population): {root}")
            # Optional: clear dirs to prevent descending further
            dirs[:] = []
            continue
        
        for f in files:
            if f.lower().endswith(('.tif', '.tiff')):
                full_path = os.path.join(root, f)
                # Split path components to extract hierarchy
                rel_path = os.path.relpath(full_path, base_day_folder)
                parts = rel_path.split(os.sep)

                #Expect at least 4 levels: cell / protocol / trial_folder / file
                if len(parts) < 4:
                   print(f"⚠️ Skipping {full_path}, unexpected structure.")
                   continue

                cell, protocol, trial_folder = parts[0], parts[1], parts[2]

                # Initialize nested dicts
                data_dict.setdefault(cell, {})
                data_dict[cell].setdefault(protocol, {})
                
                if "sd70-10" in root.lower():
                    sd70_path = full_path

                data_dict[cell][protocol][trial_folder] = tifffile.imread(full_path)

                # Load TIFF file
                print(f"📂 Loading: {cell} | {protocol} | {trial_folder} | {f}")
                data = tifffile.imread(full_path)

                # Store under structured key
                data_dict[cell][protocol][trial_folder] = data
            
    # if visualize_unexpected and unexpected_files and len(parts) < 4:
    #     print(f"\n🧩 Visualizing {len(unexpected_files)} unexpected TIFF(s)...")
    #     for path in unexpected_files:
    #         try:
    #             data = tifffile.imread(path)
    #             #plt.figure(figsize=(5, 5))
    #             if data.ndim == 3:  # assume (frames, y, x)
    #                 mean_img = np.mean(data, axis=0)
    #                 #plt.imshow(mean_img, cmap='gray')
    #                 plt.title(f"Unexpected: {os.path.basename(path)} (mean projection)")
    #             elif data.ndim == 2:
    #                 #plt.imshow(data, cmap='gray')
    #                 plt.title(f"Unexpected: {os.path.basename(path)}")
    #             else:
    #                 print(f"⚠️ Unknown data shape for {path}: {data.shape}")
    #             plt.axis('off')
    #             plt.show()
    #         except Exception as e:
    #             print(f"❌ Error loading {path}: {e}")


    return data_dict, unexpected_files

