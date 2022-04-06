# simulated annealing search of a one-dimensional objective function
import os
import numpy as np
import pandas as pd
from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed

from src.control import RL

control = RL(w=100,m=1,a=4)
data_path = '/home/lizano/Documents/SAC/data/raw/cnn/unclassified_raw_data'
cnn_data = pd.read_csv(os.path.join(data_path,'train_cnn_labels.csv'),
                        index_col=0)

def get_purity(cluster_data):
    '''
    Function that evaluates metric for clusters
    '''
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
    
    glob_purity = np.mean(maximums)
    min_purity = np.min(maximums)
    return 1/glob_purity, 1/min_purity

def transform_values(x):
    '''
    Ensure hyperparameters are inside the required range.
    '''

    if x[0] <= 2.0:
        x[0] = int(2.0)
    if x[1] <= 1.0:
        x[1] = int(1.0)
    if x[2] < 0:
        x[2] = float(0)
    elif x[2] > 1.0:
        x[2] = float(1)
    else:
        pass
    
    x[0] = int(np.around(x[0],0))
    x[1] = int(np.around(x[1],0))
    x[2] = float(x[2])

    return x


def objective(x,*args):
    '''
    function to optimize
    '''
    cluster_data = pd.read_csv(os.path.join(data_path,'hdbscan_clusters.csv'),index_col=0)
    obj = 0
    inpurity, min_inpurity, noise_size_factor = 0, 0, 0
    noise = 0
    size = 0
    control.cluster_hdbscan(mcs=int(x[0]),
                    ms=int(x[1]),
                    eps=float(x[2]))

    inpurity, min_inpurity = get_purity(cluster_data)
    noise = len(cluster_data.loc[cluster_data.labels == -1])
    size = max(cluster_data.labels.unique())+1
    noise_size_factor = ( noise + size )/len(cluster_data)
    obj = min_inpurity + noise_size_factor

    print('Gloabl Purity: ', 1/inpurity)
    print('Minimun Local Purity: ', 1/min_inpurity)
    print('Noise: ',noise)
    print('Clusters: ',size)
    print('Objective Evaluation: ',obj)

    return obj

# simulated annealing algorithm
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    print('##############Begin Simulated Annealing Optimization#################')
    # generate an initial point
    best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    best = transform_values(best)
    # evaluate the initial point
    best_eval = objective(best)
    # current working solution
    curr, curr_eval = best, best_eval
    print('Initial guess: %s, evaluation: %.5f', (best,best_eval))
    # run the algorithm
    for i in range(n_iterations):
        # take a step
        candidate = curr + randn(len(bounds)) * step_size
        # evaluate candidate point
        candidate = transform_values(candidate)
        candidate_eval = objective(candidate)
        print('--Iteration: %d, Hyperparameters: %s' %(i,candidate))
        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # report progress
            print('>%d f(%s) = %.5f' % (i, best, best_eval))
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = exp(-diff / t)
        # check if we should keep the new point
        if diff < 0 or rand() < metropolis:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval
    return [best, best_eval]

# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = asarray([[10.0, 100.0],
                    [1.0,10.0],
                    [0.0,1.0]])
# define the total iterations
n_iterations = 1000
# define the maximum step size
step_size = asarray([10.0,1.0,0.1])
# initial temperature
temp = 0.1
# perform the simulated annealing search
best, score = simulated_annealing(objective, bounds, n_iterations, step_size, temp)
print('Done!')
print('f(%s) = %f' % (best, score))