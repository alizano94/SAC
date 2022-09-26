import os
import numpy as np
import pandas as pd 
from src.control import RL
import matplotlib.pyplot as plt

control = RL(w=100,m=1,a=4)
control.createCNN()
control.loadCNN(None)

clusters_path = '/home/lizano/Documents/SAC/data/raw/cnn/clusters'

initial_clusters = pd.read_csv('/home/lizano/Documents/SAC/data/raw/cnn/UMAP-3D-clusters.csv',index_col=0)
print(initial_clusters.head())

for cluster in os.listdir(clusters_path):
    hist = [0,0,0]
    cluster_path = os.path.join(clusters_path,cluster)
    for image in os.listdir(cluster_path):
        img = os.path.join(cluster_path,image)
        state, _ = control.runCNN(img)
        hist[state] += 1
    hist /= np.sum(hist)
    if cluster != '-1' and np.max(hist) !=1:
        print('#####Cleaning cluster ',cluster,'########')
        cluster_data = initial_clusters.loc[initial_clusters['labels']==int(cluster)]
        cluster_data.drop(inplace=True,columns=['labels'])
        cluster_data.reset_index(drop=True,inplace=True)
        print(cluster_data.head())
        dump_path = os.path.join(clusters_path,cluster)
        data_path = os.path.join(clusters_path,cluster)
        control.cluster_hdbscan(data=cluster_data,df_flag=True)
        control.createCNN_DS(file='out_clusters.csv',data_path=data_path,dump_path=dump_path,delete=False)
