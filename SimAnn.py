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
    print('Sampling parameters: ',x)
    control.cluster_hdbscan(mcs=int(x[0]),
                    ms=int(x[1]),
                    eps=float(x[2]))
    
    return get_purity()

# simulated annealing algorithm
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    # generate an initial point
    best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    best = transform_values(best)
    # evaluate the initial point
    best_eval = objective(best)
    # current working solution
    curr, curr_eval = best, best_eval
    # run the algorithm
    for i in range(n_iterations):
        # take a step
        candidate = curr + randn(len(bounds)) * step_size
        # evaluate candidate point
        candidate = transform_values(candidate)
        candidate_eval = objective(candidate)
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
bounds = asarray([[2.0, 100.0],
                    [1.0,10.0],
                    [0.0,1.0]])
# define the total iterations
n_iterations = 1000
# define the maximum step size
step_size = asarray([10.0,5.0,0.1])
# initial temperature
temp = 0.1
# perform the simulated annealing search
best, score = simulated_annealing(objective, bounds, n_iterations, step_size, temp)
print('Done!')
print('f(%s) = %f' % (best, score))