import numpy as np
import matplotlib.pyplot as plt
from src.control import RL

initial_image = '/home/lizano/Documents/SAC/data/initialstates/Crystal_test.png'

control = RL(w=100,m=1,a=4)
control.umap(n=100)
control.cluster_hdbscan(n=100)
control.createCNN_DS('UMAP-100D-clusters.csv')