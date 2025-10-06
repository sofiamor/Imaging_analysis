import pandas as pd
import numpy as np
import os

tiff_path = 'F:/histed/histed_21.1_test2_n2_200msSTIM/histed_21.1_test2_n2_200msSTIM/histed_21.1_test2_n2_200msSTIM_MMStack.ome.tif'
base_name = os.path.basename(tiff_path)
file_name_without_ext = os.path.splitext(base_name)[0]
print(file_name_without_ext)

# Load the fluorescence data from the CSV
fluorescence_df = pd.read_csv('F:/histed/Traces/histed_21.1_test2_n2_200msSTIM_MMStack.ome_traces.csv')

# Define baseline window for calculating F₀ (e.g., first 50 frames)
baseline_window = 1

# Initialize a DataFrame to store ΔF/F₀ values
df_f0_df = pd.DataFrame()

# For each ROI, calculate the ΔF/F₀ time series
for column in fluorescence_df.columns:
    # Get the fluorescence data for the current ROI
    roi_fluorescence = fluorescence_df[column].values
    
    # Calculate the baseline (F₀) for this ROI, using the first `baseline_window` frames
    F0 = np.percentile(roi_fluorescence[:baseline_window], 10)  # 10th percentile as baseline
    print(f"Baseline (F₀) for {column}: {F0}")
    
    # Calculate ΔF/F₀ for each frame
    df_f0 = (roi_fluorescence - F0) / F0
    
    # Store the ΔF/F₀ values in the DataFrame
    df_f0_df[column] = df_f0

# Save the ΔF/F₀ values to a CSV
output_df_path = f'F:/histed/df_over_f0/{file_name_without_ext}_df_f0_traces.csv'
df_f0_df.to_csv(output_df_path, index=False)

print(f"ΔF/F₀ traces saved to {output_df_path}")
