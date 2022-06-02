import os

from src.stateRep import CNN_Testing

cnn = CNN_Testing()
cnn.createCNN()
#cnn.trainCNN(batch=10,
#            epochs=8,
#            plot=True)
cnn.loadCNN(None)
cnn.testCNN(None)
