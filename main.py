import os
from src.control import RL

full_features_csv = "full_raw_features.csv"
projected_features_csv = "full_umap_2D_features.csv"
clusters_csv = 'full_'

control = RL(w=100,m=1,a=4)
control.createCNN()
control.trainCNN(plot=True)
