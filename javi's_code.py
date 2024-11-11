import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join, basename, splitext
import datetime

now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

#this code creates a list with all the files that exist in a folder
traces_path = r"Z:/smorou/Analysis/imaging_analysis/CalciumIm/Traces/"
coords_path = r"Z:/smorou/Analysis/imaging_analysis/CalciumIm/Coordinates"
tiff_path = r"Z:/smorou/Calcium Imaging/FV1/FV1 SPONT/FV1 SPONT_MMStack.ome.tif"

base_name = basename(tiff_path)
file_name_without_ext = splitext(base_name)[0]
print(f"Processing file: {file_name_without_ext}")

traces_list = [f for f in listdir(traces_path)]
coords_list = [f for f in listdir(coords_path) if isfile(join(coords_path,f))]

traces, coords = [], []

for i in range(len(traces_list)):
    file_path = join(traces_path, traces_list[i])
    temp_traces = pd.read_csv(file_path, encoding='utf-8', lineterminator='\n') #, error_bad_lines=False)
    cells_list = list(temp_traces.keys())
    cells_list = cells_list[0:]
    temp = []
    for j in cells_list:
        temp.append(temp_traces[j])


    temp_coords = []
    file_path = join(coords_path, coords_list[i])
    temp_coordss = pd.read_csv(coords_path +'/' + coords_list[i], encoding='utf-8', lineterminator='\n') #, error_bad_lines=False)
    temp_coords = temp_coordss.values.tolist()
    temp = [t.tolist() for t in temp]
    traces.append(temp[50:350])
    coords.append(temp_coords)


coordinates = np.asarray(coords[0])

data = np.asarray(traces[0]).T


fig, axs = plt.subplots(3, 1, sharey=True)
axs[0].plot(data[:2], color='black')
axs[1].plot(data[:3], color='darkgrey')
axs[2].plot(data[:4], color='grey')
for ax in axs:
    ax.tick_params(axis='x')#, colors='white')
    ax.tick_params(axis='y')#, colors='white')
fig.suptitle('Example cells')#, color='white')

plt.tight_layout()
file_name = f"Z:/smorou/Analysis/imaging_analysis/CalciumIm/Plots/{file_name_without_ext}{timestamp}.png"
additional_string = '_example_cells'
plt.savefig(f"{file_name.replace('.png', '')}{additional_string}.png", transparent=True)

plt.figure()
for i in range(len(data.T)):
    plt.subplot(len(data.T),1,i+1)
    plt.plot(data[:,i],c=(np.random.rand(3)))
    plt.yticks([])

plt.tick_params(axis='x')#, colors='white')
plt.suptitle('All cells')#, color='white')
plt.xlabel('Time')#, color='white')
additional_string1 = '_all_cells'
plt.savefig(f"{file_name.replace('.png', '')}{additional_string1}.png", transparent=True)

corr_values = np.empty( (len(data.T), len(data.T)) )
plt.figure()
for i in range(len(data.T)):
    for j in range(0,len(data.T)):
        temp = np.corrcoef(data[:,i],data[:,j])
        corr_values[i,j] = temp[0,1]

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.figure()
plt.imshow(corr_values)
plt.title('Correlation matrix',color='white')
plt.colorbar()
plt.tick_params(axis='x')#, colors='white')
plt.tick_params(axis='y')#, colors='white')
additional_string2 = '_corr_matrix'
plt.savefig(f"{file_name.replace('.png', '')}{additional_string2}.png", transparent=True)

# Ensure data is a 2-dimensional array
print('data shape:', data.shape)

corr_index = np.zeros((len(data.T)))

# Check if data is 2-dimensional
if len(data.shape) != 2:
    raise ValueError("data should be a 2-dimensional array")

reordered_data = data[:, np.argsort(corr_index)]

for j in range(0, len(data.T)):
    temp = np.corrcoef(data[:,0],reordered_data[:,j])
    corr_index[j] = temp[0,1]

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.figure()
plt.imshow(corr_values)
plt.title('Reordered Corr. matrix')
plt.colorbar()

reordered_coords = coordinates[np.argsort(corr_index),:].astype(int)
reordered_data = data[:, np.argsort(corr_index)].T
print('reordered_data', reordered_data.shape)
print('reordered_coords', reordered_coords.shape)

plt.figure()

#connection of cell activity, scatter and line 
for i in range(0,len(data.T)):
    plt.scatter(reordered_coords[i,1],reordered_coords[i,0])
plt.title('Cell positions', color='white')
plt.tick_params(axis='x')#, colors='white')
plt.tick_params(axis='y')#, colors='white')
additional_string3 = '_cell_positions'
plt.savefig(f"{file_name.replace('.png', '')}{additional_string3}.png", transparent=True)

plt.figure()

for i in range(len(data.T)):
    for j in range(len(data.T)):
        #print(reordered_coords[i,:],reordered_data[:,j])
        temp = np.corrcoef(reordered_data[:,i],reordered_data[:,j])
        plt.plot([reordered_coords[i,0], reordered_coords[j,0]], [reordered_coords[i,1], reordered_coords[j,1]], color='white', linewidth = temp[0,1])
plt.title('Correlation plot of cells')#, color='white')
plt.tick_params(axis='x')#, colors='white')
plt.tick_params(axis='y')#, colors='white')
additional_string4 = '_correl_line'
plt.savefig(f"{file_name.replace('.png', '')}{additional_string4}.png", transparent=True)

#graph for all vids
plt.figure()
all_correlations = []
for cell in range(len(traces)):
    coordinates = np.asarray(coords[cell]).astype(int)
    data = np.asarray(traces[cell]).T
    corr_values = np.empty((len(data.T), len(data.T)))

    for i in range(0,len(data.T)):
        for j in range(0,len(data.T)):
            temp = np.corrcoef(data[:,i], data[:,j])
            corr_values[i,j] = temp[0,1]

    ax = plt.subplot(6,4,cell+1)

    for i in range(0,len(data.T)):
        for j in range(0,len(data.T)):
            if corr_values[i,j]>0.9:
                plt.plot([reordered_coords[i,0], reordered_coords[j,1]], [reordered_coords[i,1], reordered_coords[j,1]], 'k', linewidth = temp[0,1])
    
    for i in range(0,len(data.T)):
        plt.scatter(coordinates[i,0],coordinates[i,1],s=35)

    all_correlations.append(corr_values)

plt.title(traces_list[cell])
ax.axis('off')

# import networkx as nx
# import community as community_louvain
# import community.community_louvain as community_louvain


# # Create a simple graph
# G = nx.karate_club_graph()

# # Find the best partition
# partition = community_louvain.best_partition(G)

# # Verify the structure of the partition
# print(partition)

# # Calculate the modularity
# modularity = nx.community.modularity(G, partition)

# print("Modularity:", modularity)


# modularity_values =np.zeros((len(all_correlations)))
# degree_values =np.zeros((len(all_correlations)))
# clustering_values =np.zeros((len(all_correlations)))

# for num in range(0,len(all_correlations)):
#     correlation_matrix = all_correlations[num]
#     threshold =0.9
#     adjacency_matrix = np.zeros((np.shape(correlation_matrix)))
#     for i in range(0,len(correlation_matrix)):
#         for j in range(0,len(correlation_matrix)):
#             if correlation_matrix[i,j]>threshold:
#                 adjacency_matrix[i,j] = 1


#     G = nx.Graph(adjacency_matrix)
#     # Compute the best partition using the Louvain method
#     partition = community_louvain.best_partition(G)
#     modularity = nx.community.modularity(partition, G)
#     modularity_values[num] = modularity

#     degree = np.sum(adjacency_matrix, axis =1)
#     degree_values =np.mean(degree)

#     clustering = nx.clustering(G)
#     clustering_values[num] = np.mean(list(clustering.values()))

# plt.figure()
# plt.scatter(modularity_values,degree_values)