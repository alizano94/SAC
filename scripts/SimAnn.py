import os
import numpy as np
from scipy import optimize
from src.control import RL

class MyBounds:
    def __init__(self, xmax=[100.0,10.0,1.0], xmin=[1.0,1.0,0.0] ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

np.random.seed(555)
control = RL(w=100,m=1,a=4)
control.createCNN()
control.loadCNN(None)

def get_purity():
    '''
    Function that evaluates metric for clusters
    '''
    clusters_path = '/home/lizano/Documents/SAC/data/raw/cnn/clusters'
    purities = []
    for cluster in os.listdir(clusters_path):
        if cluster != '-1':
            hist = [0,0,0]
            cluster_path = os.path.join(clusters_path,cluster)
            for image in os.listdir(cluster_path):
                img = os.path.join(cluster_path,image)
                state, _ = control.runCNN(img)
                hist[state] += 1
            hist /= np.sum(hist)
            purities.append(np.max(hist))
    purity = np.mean(purities)
    return 1/purity

def transform_values(x):
    '''
    Ensure hyperparameters are inside the required range.
    '''
    mcs = int(x[0])
    ms = int(x[1])
    eps = x[2]

    if mcs <= 0:
        mcs = 1
    if ms <= 0:
        ms = 1
    
    if eps < 1.0:
        eps = 1.0
    elif eps > 0.0:
        eps = 0.0
    else:
        pass

    return mcs, ms, eps


def Obj(x,*args):
    '''
    function to optimize
    '''
    print('Sampling values: ',x)
    mcs, ms, eps = transform_values(x)
    control.cluster_hdbscan(mcs=mcs,
                    ms=ms,
                    eps=eps)
    control.createCNN_DS()
    print('Obj Evaluation Done!')
    
    return get_purity()

x0 = np.array([np.random.randint(1,1000),
                np.random.randint(1,100),
                np.random.uniform(0.0,1.0)])

minimizer_kwargs = {'method': 'L-BFGS-B'}
mybounds = MyBounds()
res = optimize.basinhopping(Obj, x0, minimizer_kwargs=minimizer_kwargs,
                            disp=True,niter=1000)

print(res)


