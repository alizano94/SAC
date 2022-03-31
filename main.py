import numpy as np
import matplotlib.pyplot as plt
from src.control import RL

control = RL(w=100,m=1,a=4)
control.raw_featInception()
control.umap(n=3)
control.cluster_hdbscan(mcs=8,ms=1,eps=float(0.39348044496354423))
control.createCNN_DS()
