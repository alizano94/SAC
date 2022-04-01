import numpy as np
import matplotlib.pyplot as plt
from src.control import RL

control = RL(w=100,m=1,a=4)
control.cluster_hdbscan(mcs=10,ms=6,eps=float(0.0))
control.createCNN_DS()
