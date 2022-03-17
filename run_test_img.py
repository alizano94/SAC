import numpy as np
import matplotlib.pyplot as plt
from src.control import RL

initial_image = '/home/lizano/Documents/SAC/data/initialstates/Crystal_test.png'

control = RL(w=100,m=1,a=4)
img_batch = control.preProcessImg(initial_image,IMG_H=848,IMG_W=848)
plt.imshow(img_batch[0])
plt.gray()
plt.axis('off')
plt.show()
plt.clf()
