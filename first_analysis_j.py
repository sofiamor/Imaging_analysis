import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join 
import networkx as nx


#this code creates a list with all the files that exist in a folder
traces_path = r"C:/Users/sofik/Desktop/analysis/Traces"
coords_path = r"C:/Users/sofik/Desktop/analysis/Coordinates"

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
print('data', data)

plt.figure()
plt.plot(data[:,0])
plt.title('Cell 1')


plt.figure()
for i in range(len(data.T)):
    plt.subplot(len(data.T),1,i+1)
    plt.plot(data[:,i],c=(np.random.rand(3)))

corr_values = np.empty( (len(data.T), len(data.T)) )

for i in range(len(data.T)):
    for j in range(0,len(data.T)):
        temp = np.corrcoef(data[:,i],data[:,j])
        corr_values[i,j] = temp[0,1]

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.figure()
plt.imshow(corr_values)
plt.title('Correlation matrix')
plt.colorbar()

corr_index = np.empty((len(data.T)))
reordered_data = data[:, np.argsort(corr_index)]

for j in range(0,len(data.T)):
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
plt.title('Scatter plot of cells')

plt.figure()

for i in range(len(data.T)):
    for j in range(len(data.T)):
        #print(reordered_coords[i,:],reordered_data[:,j])
        temp = np.corrcoef(reordered_data[:,i],reordered_data[:,j])
        plt.plot([reordered_coords[i,0], reordered_coords[j,0]], [reordered_coords[i,1], reordered_coords[j,1]], 'k', linewidth = temp[0,1])
plt.title('Correlation plot of cells')

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

modularity_values =np.zeros((len(all_correlations)))
degree_values =np.zeros((len(all_correlations)))
clustering_values =np.zeros((len(all_correlations)))

for num in range(0,len(all_correlations)):
    correlation_matrix = all_correlations[num]
    threshold =0.9
    adjacency_matrix = np.zeros((np.shape(correlation_matrix)))
    for i in range(0,len(correlation_matrix)):
        for j in range(0,len(correlation_matrix)):
            if correlation_matrix[i,j]>threshold:
                adjacency_matrix[i,j] = 1


    G = nx.Graph(adjacency_matrix)
    modularity = nx.community.modularity(G, nx.community.asyn_lpa_communities(G))
    modularity_values[num] = modularity

    degree = np.sum(adjacency_matrix, axis =1)
    degree_values =np.mean(degree)

    clustering = nx.clustering(G)
    clustering_values[num] = np.mean(list(clustering.values()))

plt.figure()
plt.scatter(modularity_values,degree_values)