import os

from src.control import RL


control = RL(w=100,m=1,a=4)
control.createCNN_DS()
control.createCNN_spits(testing_split=0.1,
                        validation_split=0.1)