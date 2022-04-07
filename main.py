import numpy as np
import matplotlib.pyplot as plt
from src.control import RL

control = RL(w=100,m=1,a=4)
control.createCNN(summary=True)
control.trainCNN(batch=5,epochs=60)
