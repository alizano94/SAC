import os 
import pandas as pd
import matplotlib.pyplot as plt

cluster_file_path = os.path.join(os.getcwd(),'..')
cluster_file_path = os.path.join(cluster_file_path,'data/raw/cnn/unclassified_raw_data')
cluster_file_path = os.path.join(cluster_file_path,'hdbscan_clusters.csv')

clusters = pd.read_csv(cluster_file_path, index_col=0)

x = clusters['UMAP 0'].to_numpy()
y = clusters['UMAP 1'].to_numpy()
z = clusters['UMAP 2'].to_numpy()
c = clusters['labels'].to_numpy()

# Creating figure
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
   
# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)
 
 
# Creating color map
my_cmap = plt.get_cmap('brg')
 
# Creating plot
sctt = ax.scatter3D(x, y, z,
                    alpha = 0.8,
                    c = c,
                    cmap = my_cmap,
                    marker ='^')
 
plt.title("simple 3D scatter plot")
ax.set_xlabel('UMAP 0', fontweight ='bold')
ax.set_ylabel('UMAP 1', fontweight ='bold')
ax.set_zlabel('UMAP 2', fontweight ='bold')
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
 
# show plot
plt.show()