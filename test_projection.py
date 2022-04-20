import os
from src.dataPrep import Clustering_Test

test = Clustering_Test()
print(test.pca_reconstruction_error(n_samples=500))