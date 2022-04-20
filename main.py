import os
from src.control import RL

control = RL(w=100,m=1,a=4)

path = os.path.join(control.cnn_ds_path,'unclassified_raw_data','validation')
