import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

_ = get_purity()