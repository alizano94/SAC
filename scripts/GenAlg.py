import os
import numpy as np
import pandas as pd
from scipy import optimize

from src.control import RL

np.random.seed(555)
control = RL(w=100,m=1,a=4)
data_path = '/home/lizano/Documents/SAC/data/raw/cnn/unclassified_raw_data'
cnn_data = pd.read_csv(os.path.join(data_path,'train_cnn_labels.csv'),
                        index_col=0)

def get_purity():
    '''
    Function that evaluates metric for clusters
    '''
    cluster_data = pd.read_csv(os.path.join(data_path,'hdbscan_clusters.csv'))
    purities = np.zeros((np.max(cluster_data.labels.unique())+1,3))
    for i in range(len(cluster_data)):
        row = cluster_data.loc[i,'labels']
        column = 0
        for j in range(len(cnn_data)):
            if cluster_data.loc[i,'Image Names'] == cnn_data.loc[j,'Image Names']:
                column = cnn_data.loc[j,'CNN Labels']
        purities[row][column] += 1

    for i in range(len(purities)):
        sum = np.sum(purities[i])
        for j in range(len(purities[i])):
            purities[i][j] /= sum

    maximums = []
    for i in range(len(purities)):
        maximums.append(np.max(purities[i]))
    
    purity = np.mean(maximums)
    print('Purity: ',purity)
    print('Inpurity: ',1/purity)
    return 1/purity

def transform_values(x):
    '''
    Ensure hyperparameters are inside the required range.
    '''
    mcs = int(x[0])
    ms = int(x[1])
    eps = float(x[2])

    if mcs <= 0:
        mcs = 1
    if ms <= 0:
        ms = 1
    

    return mcs, ms, eps


def Obj(x,*args):
    '''
    function to optimize
    '''
    
    mcs, ms, eps = transform_values(x)
    print('Sampling values: ',[mcs,ms,eps])
    control.cluster_hdbscan(mcs=mcs,
                    ms=ms,
                    eps=eps)
    
    return get_purity()

bounds = [(1,100),(1,10),(0,1)]

res = optimize.differential_evolution(Obj, bounds, disp = True)

print(res)


