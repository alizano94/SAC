import os
from src.dataPrep import Clustering_Test

test = Clustering_Test()
components = [2,3,10,100,200,300,400,500]
for n in components:
    MSE = test.umap_reconstruction_error(n=n,n_samples=600)
    print('Components: ',str(n), 'MSE: ',str(MSE))