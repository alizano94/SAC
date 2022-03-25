import numpy as np
import matplotlib.pyplot as plt
from src.control import RL

initial_image = '/home/lizano/Documents/SAC/data/initialstates/Crystal_test.png'

control = RL(w=100,m=1,a=4)
control.umap(n=2)
control.cluster_hdbscan(n=2)
control.createCNN_DS('UMAP-2D-clusters.csv')