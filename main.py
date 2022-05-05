import os

from numpy import full
from src.control import RL

full_features_csv = "full_raw_features.csv"
projected_features_csv = "full_umap_2D_features.csv"
clusters_csv = 'full_hdbscan_UMAP2.csv'

control = RL(w=100,m=1,a=4)
control.createCNN_DS(load_file='full_hdbscan_UMAP2.csv')