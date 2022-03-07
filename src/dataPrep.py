import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

from src.helpers import Helpers

class IMG_Clustering(Helpers):
    def __init__(self,k, *args, **kwargs):
        super(IMG_Clustering, self).__init__(*args, **kwargs)
        self.k = k

    def feature_extractor(self):
        '''
        Method that takes a dump of images and extracts their features
        using the InceptionV3 model.
        args:
        returns:
        '''
        model = InceptionV3(weights='imagenet', include_top=False)
        features = []
        img_name = []
        img_path = os.listdir(direc)
        for i in tqdm(img_path):
            fname=direc+'/'+i
            img=image.load_img(fname,target_size=(224,224))
            x = img_to_array(img)
            x=np.expand_dims(x,axis=0)
            x=preprocess_input(x)
            feat=model.predict(x)
            feat=feat.flatten()
            features.append(feat)
            img_name.append(i)
        columns_names = ['Image Name']
        columns_feat = []
        for i in range(len(features[0])):
            header = 'raw '+str(i)
            columns_feat.append(header)
