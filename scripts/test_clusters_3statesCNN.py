import os
import numpy as np
from src.control import RL
import matplotlib.pyplot as plt

control = RL(w=100,m=1,a=4)
control.createCNN()
control.loadCNN(None)

clusters_path = '/home/lizano/Documents/SAC/data/raw/cnn/clusters'
save_path = '/home/lizano/Documents/SAC/results/clusters'

os.system('rm -rf '+os.path.join(save_path,'*'))
hist_max = []

for cluster in os.listdir(clusters_path):
    hist = [0,0,0]
    cluster_path = os.path.join(clusters_path,cluster)
    for image in os.listdir(cluster_path):
        img = os.path.join(cluster_path,image)
        state, _ = control.runCNN(img)
        hist[state] += 1
    hist /= np.sum(hist)
    if cluster != '-1':
        print('Local purity for cluster ',cluster,': ',np.max(hist))
        hist_max.append(np.max(hist))
    outfig = 'Cluster_'+cluster+'-hist.png'
    outfig = os.path.join(save_path,outfig)
    labels = ['S0','S1','S2']
    plt.bar(np.arange(len(hist)),hist,color='black')
    plt.xlabel('3-State Classification')
    plt.title('Cluster '+cluster)
    plt.ylabel('Frequency')
    plt.xticks(np.arange(len(hist)),labels)
    #plt.show()
    figure = plt.gcf()
    figure.set_size_inches(16,8)
    plt.savefig(outfig,dpi=100)
    plt.clf()

print('Found ',len(hist_max),' clusters.')
purity = np.mean(hist_max)
print('Cluster putrity: ', purity)
