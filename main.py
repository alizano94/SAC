import numpy as np
import matplotlib.pyplot as plt
from src.control import RL

initial_image = '/home/lizano/Documents/SAC/data/initialstates/Crystal_test.png'

control = RL(w=100,m=1,a=4)
control.createCNN_DS(file='UMAP-3D-clusters.csv')