import numpy as np
import matplotlib.pyplot. as plt
import pandas as pd
from os import listdir
from os.path import isfile, join 
import networkx as nx


#this code creates a list with all the files that exist in a folder
traces_path = r"E:/.../.../...../Traces"
coords_path = r"E:/.../.../.../Coordinates"

traces_list = [f for f in listdir(traces_path)]
coords_list = [f for f in listdir(coords_path) if isfile(join(coords_path,f))]

traces, coords = [], []

for i in range(0, len(trace_list)):
    temp_traces = pd.read_csv(traces_path + '/' + traces_list[i])
    cells_list = list(temp_traces.keys())
    cells_list = cells_list[1:]
    temp = []
    for j in cells_list:
        temp.append(temp_traces[j])

    temp_coords = []
    with open(coords_path +'/' + coords_list[i], newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            temp_coords.append(row)
    traces.append(temp)
    coords.append(temp_coords)

coordinates = np.asarray(coords[0])
data = np.asarray(traces[0]).T

plt.figure()
plt.plot(data[:,0])


plt.figure()
for i in range(0, len(data.T)):
    plt.subplot(len(data.T),1,i+1)
    plt.plot(data[:,i],c=(np.random.rand(3)))

corr_values = np.empty( (len(data.T), len(data.T)) )

for i in range(0, len(data.T)):
    for j in range(0,len(data.T)):
        temp = np.corrcoef(data[:,i],data[:,j])
        corr_values[i,j] = temp[0,1]

plt.figure()
plt.imshow(corr_values)
plt.colorbar()

reordered_data = data[:, np.argsort(corr_index)]
corr_index = np.empty((len(data.T)))

for j in range(0,len(data.T)):
    temp = np.corrcoef(data[:,0],reordered_data[:,j])
    corr_index[j] = temp[0,1]

plt.figure()
plt.imshow(corr_values)
plt.colorbar()

reordered_coords = coordinates[np.agsort(corr_index),:].astype[int]

plt.figure(),

#connection of cell activity, scatter and line 
for i in range(0,len(data.T)):
    plt.scatter(reordered_coords[i,0],reordered_coords[i,1],s=100)

for i in range(0,len(data.T)):
    for j in range(0,len(data.T)):
        temp = np.corrcoef(reordered_coords[:,i],reordered_data[:,j])
        plt.plot(reordered_coords[i,0], reordered_coords[j,1], reordered_coords[i,1], reordered_coords[j,1], 'k', linewidth = temp[0,1])

#graph for all vids
plt.figure()
for cell in range(0,#number of vids#):
    coordinates = np.asarray(coords[cell]).astype(int)
    data = np.asarray(traces[cell]).T
    corr_values = np.empty((len(data.T), len(data.T)))

    for i in range(0,len(data.T)):
        for j in range(0,len(data.T)):
            temp = np.corrcoeff(data[:,i], data[:,j])
            corr_values[i,j] = temp[0,1]

    ax = plt.subplot(6,4,cell=1)

    for i in range(0,len(data.T)):
        for j in range(0,len(data.T)):
            if corr_values[i,j]>0,9:
                plt.plot(reordered_coords[i,0], reordered_coords[j,1], reordered_coords[i,1], reordered_coords[j,1], 'k', linewidth = temp[0,1])
    
    for i in range(0,len(data.T)):
    plt.scatter(coordinates[i,0],coordinates[i,1],s=35)
plt.title(traces_list[cell])
ax.axis('off')

modularity_values =np.zeros((len(all_correlations)))
degree_values =np.zeros((len(all_correlations)))
clustering_values =np.zeros((len(all_correlations)))

for num in range(0,len(all_correlations)):
    correlation_matrix = all_correlations[num]
    threshold =0.9
    adjacent_matrix = np.zeros(np.shape(correlation_matrix))
    for i in range(0,len(correlation_matrix)):
        for j in range(0,len(correlation_matrix)):
            if correlation_matrix[i,j]>threshold:
                adjacency_matrix[i,j] = 1


    G = nx.Graph(adjacency_matrix)
    modularity = nx.community.modularity(G, nx.community.lowain_communities(G))
    modularity_values[num] = modularity

    degree = np.sum(adjacency_matrix, axis =1)
    degree_values =np.mean(degree)

    clustering = nx.clustering(G)
    clustering_values[num] = np.mean(list(clustering.values()))

plt.figure()
plt.scatter(modularity_values,degree_values)