import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('/home/lizano/Documents/SAC/data/raw/cnn/dump/data_parameters.csv',
                    index_col=0)
print(data.head())

clusters_path = '/home/lizano/Documents/SAC/data/raw/cnn/clusters'
save_path = '/home/lizano/Documents/SAC/results/clusters'

labels = [-1,0,1,2,3,4,5,6,7,8,9]

mean_C6 = []
mean_psi6 = []
std_C6 = []
std_psi6 = []
for i in range(len(os.listdir(clusters_path))):
    mean_C6.append(0)
    mean_psi6.append(0)
    std_C6.append(0)
    std_psi6.append(0)
print(len(mean_C6))


for cluster in os.listdir(clusters_path):
    C6 = []
    psi6 = []
    cluster_path = os.path.join(clusters_path,cluster)
    for image in os.listdir(cluster_path):
        for i in range(len(data)):
            if image == data['name'].iloc[i]:
                C6.append(data['C6_avg'].iloc[i])
                psi6.append(data['Psi6'].iloc[i])             
    mean_C6[int(cluster)+1] = np.average(C6)
    mean_psi6[int(cluster)+1] = np.average(psi6)
    std_C6[int(cluster)+1] = np.std(C6)
    std_psi6[int(cluster)+1] = np.std(psi6)

outfig = 'Mean_Psi6.png'
outfig = os.path.join(save_path,outfig)
plt.errorbar(labels,mean_psi6,yerr=std_psi6,fmt='.k')
plt.xlabel('Cluster')
plt.title('Mean Psi 6')
plt.ylabel('Psi 6')
plt.xticks(labels,labels)
figure = plt.gcf()
figure.set_size_inches(8,8)
plt.savefig(outfig,dpi=100)
plt.clf()

outfig = 'Mean_C6.png'
outfig = os.path.join(save_path,outfig)
plt.errorbar(labels,mean_C6,yerr=std_C6,fmt='.k')
plt.xlabel('Cluster')
plt.title('Mean C 6')
plt.ylabel('C6')
plt.xticks(labels,labels)
figure = plt.gcf()
figure.set_size_inches(8,8)
plt.savefig(outfig,dpi=100)
plt.clf()

print(mean_C6)
print(mean_psi6)
print(std_C6)
print(std_psi6)