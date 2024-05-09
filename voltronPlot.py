import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join 


#this code creates a list with all the files that exist in a folder
traces_path = r"C:/Users/LabPC/Desktop/Traces"
coords_path = r"C:/Users/LabPC/Desktop/Coordinates"

traces_list = [f for f in listdir(traces_path)]
coords_list = [f for f in listdir(coords_path) if isfile(join(coords_path,f))]

traces, coords = [], []

for i in range(len(traces_list)):
    file_path = os.path.join(traces_path, traces_list[i])
    temp_traces = pd.read_csv(file_path, encoding='utf-8', lineterminator='\n', error_bad_lines=False)
    cells_list = list(temp_traces.keys())
    cells_list = cells_list[1:]
    temp = []
    for j in cells_list:
        temp.append(temp_traces[j])


    temp_coords = []
    file_path = os.path.join(coords_path, coords_list[i])
    temp_coordss = pd.read_csv(coords_path +'/' + coords_list[i], encoding='utf-8', lineterminator='\n', error_bad_lines=False)
    temp_coords = temp_coordss.values.tolist()
    temp = [t.tolist() for t in temp]
    traces.append(temp)
    coords.append(temp_coords)


coordinates = np.asarray(coords[0])

data = np.asarray(traces[0]).T


fig, axs = plt.subplots(3, 1, sharey=True)
axs[0].plot(data[:,2], color='white')
axs[1].plot(data[:,3], color='darkgrey')
axs[2].plot(data[:,4], color='grey')
for ax in axs:
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
fig.suptitle('Example cells', color='white')

plt.tight_layout()
plt.savefig('C:/Users/sofik/Desktop/3cells.png', transparent=True)
