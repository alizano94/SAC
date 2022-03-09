import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
from umap import UMAP

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
        args:None
        returns:
            -data: DataFrame containing the raw features.
        '''
        direc = os.path.join(self.cnn_ds_path,'dump')
        model = InceptionV3(weights='imagenet', include_top=False)
        raw_features = []
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
            raw_features.append(feat)
            img_name.append(i)
        columns_names = ['Image Name']
        columns_feat = []
        for i in range(len(raw_features[0])):
            header = 'raw '+str(i)
            columns_feat.append(header)
        img_name,raw_features = np.row_stack(img_name),np.row_stack(raw_features)
        img_name,raw_features = pd.DataFrame(img_name,columns=columns_names), pd.DataFrame(raw_features,columns=columns_feat)
        data = pd.concat([img_name,raw_features],axis=1,join='inner')
        print(data.head())

        return data

    def tSNE(self,n):
        '''
        Performs tSNE reduction on the raw features for the images.
        args:
            -n: number of componets to reduce to
        return: None
        '''

        data = self.feature_extractor()
        out_name = 'tSNE-'+str(n)+'components-features.csv'

        raw_features = data.drop(columns=['Image Name'])
        image_names = data.pop('Image Name')

        raw_features.to_numpy()
        columns = []
        for i in range(n):
            header = 'tSNE '+str(i)
            columns.append(header)
        tSNE = TSNE(n_components=n)
        features = tSNE.fit_transform(raw_features)
        features = pd.DataFrame(features,columns=columns)
        data = pd.concat([image_names,features],axis=1,join='inner')
        print(data.head())
        data.to_csv(os.path.join(self.cnn_ds_path,out_name))

    def umap(self,n):
        '''
        Performs tSNE reduction on the raw features for the images.
        args:
            -n: number of componets to reduce to
        return: None
        '''

        data = self.feature_extractor()
        out_name = 'UMAP-'+str(n)+'components-features.csv'

        raw_features = data.drop(columns=['Image Name'])
        image_names = data.pop('Image Name')

        raw_features.to_numpy()
        columns = []
        for i in range(n):
            header = 'UMAP '+str(i)
            columns.append(header)
        umap_3d = UMAP(n_components=n)
        features = umap_3d.fit_transform(raw_features)
        features = pd.DataFrame(features,columns=columns)
        data = pd.concat([image_names,features],axis=1,join='inner')
        print(data.head())
        data.to_csv(os.path.join(self.cnn_ds_path,out_name))

