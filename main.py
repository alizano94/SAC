import os

from src.control import RL


control = RL(w=100,m=1,a=4)
#control.raw_featInception()
#control.umap(
#    n_components=3,
#    n_neighbors=500,
#    min_dist=0.0121
#)
control.cluster_hdbscan(
    mcs=200,
    ms=50,
    eps=0.25
)
control.createCNN_DS()
control.createCNN_spits(
    testing_split=0.1,
    validation_split=0.1
)