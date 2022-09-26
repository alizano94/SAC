import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SMALL_SIZE = 24
MEDIUM_SIZE = 32
BIGGER_SIZE = 64

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


data_path = '/home/lizano/Documents/SAC/data/raw/cnn/unclassified_raw_data'
cnn_data = pd.read_csv(os.path.join(data_path,'full_cnn_labels.csv'),
                        index_col=0)
results_path = '/home/lizano/Documents/SAC/results/clusters'
os.system('rm -rf '+os.path.join(results_path,'*'))

def get_purities():
    '''
    Function that evaluates metric for clusters
    '''
    cluster_data = pd.read_csv(os.path.join(data_path,'full_hdbscan_UMAP500.csv'))
    purities = np.zeros((np.max(cluster_data['labels'].unique())+1,3))
    for i in range(len(cluster_data)):
        row = cluster_data.loc[i,'labels']
        column = 0
        for j in range(len(cnn_data)):
            if cluster_data.loc[i,'Image Names'] == cnn_data.loc[j,'Image Names']:
                column = cnn_data.loc[j,'CNN Labels']
        purities[row][column] += 1

    
    return purities
    
purities = get_purities()

for i in range(len(purities)):
    hist = purities[i]
    x = np.arange(len(hist))
    plt.bar(x,height=hist,color='black')
    xticks = []
    for j in range(len(hist)):
        name = 'S'+str(j)
        xticks.append(name)
    plt.xticks(x,xticks)
    figure = plt.gcf()
    figure.set_size_inches(16,9)
    save_name = 'Cluster'+str(i)+'_dist.png'
    plt.savefig(os.path.join(results_path,save_name),dpi=100)
    plt.clf()