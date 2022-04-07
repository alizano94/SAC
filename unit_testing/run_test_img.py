import numpy as np
import matplotlib.pyplot as plt
from src.control import RL

plot = True
initial_image = '/home/lizano/Documents/SAC/data/initialstates/Defective1.png'


control = RL(w=100,m=1,a=4)
control.createAE()
control.loadAE(None)
img_batch = control.preProcessImg(initial_image)
print(img_batch)
if plot:
    plt.imshow(img_batch[0])
    plt.gray()
    plt.axis('off')
    plt.show()
    plt.clf()

reconstructed = 255*control.autoencoder.predict(img_batch)
print(reconstructed)
if plot:
    plt.imshow(reconstructed[0])
    plt.gray()
    plt.axis('off')
    plt.show()
    plt.clf()
