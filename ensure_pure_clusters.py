import os
import numpy as np
import pandas as pd 
#from src.control import RL
import matplotlib.pyplot as plt

data = pd.read_csv('/home/lizano/Documents/SAC/data/raw/cnn/UMAP-3D-clusters.csv')
data.drop(columns=['Unnamed: 0','Unnamed: 0.1'],inplace=True)
print(data.head())
