import os
import pandas as pd 

from src.control import RL

control = RL(w=100,m=1,a=4)
control.createCNN()
control.loadCNN(None)


data_path = '/home/lizano/Documents/SAC/data/raw/cnn/unclassified_raw_data/train'

data = pd.DataFrame(columns=['Image Names','CNN Labels'])

names = []
states = []

for image in os.listdir(data_path):
    img = os.path.join(data_path,image)
    state, _ = control.runCNN(img)
    names.append(image)
    states.append(state)

data['Image Names'] = names
data['CNN Labels'] = states

data_path = '/home/lizano/Documents/SAC/data/raw/cnn/unclassified_raw_data'
data.to_csv(os.path.join(data_path,'train_cnn_labels.csv'))