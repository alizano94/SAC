import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
from umap import UMAP
import hdbscan
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from src.helpers import Helpers

class AutoEncoder(Model,Helpers):
    def __init__(self,latent_dim,IMG_H=212,IMG_W=212, chan=1,*args,**kwargs):
        super(AutoEncoder,self).__init__(*args,**kwargs)
        self.latent_dim = latent_dim
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.chan = chan

        self.encoder = Sequential([
            layers.Input(shape=(self.IMG_H, self.IMG_W, self.chan)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(self.latent_dim, activation='relu')])

        self.decoder = Sequential([
            layers.Dense(self.IMG_H*self.IMG_W*self.chan, activation = 'relu', input_shape=(self.latent_dim,)),
            layers.Reshape((self.IMG_H, self.IMG_W, self.chan)),
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class IMG_Clustering(Helpers):
    def __init__(self, *args, **kwargs):
        super(IMG_Clustering, self).__init__(*args, **kwargs)
        self.AE_save_path = os.path.join(self.cnn_weights_path,'AE.h5')

    def raw_featInception(self):
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

    def createAE(self,latent_dim=1000,summary=False):
        self.autoencoder = AutoEncoder(latent_dim)
        self.autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        if summary:
            self.autoencoder.summary()
            tf.keras.utils.plot_model(
				model = self.autoencoder,
				rankdir="TB",
				dpi=72,
				show_shapes=True
				)

    def trainAE(self,batch=32,epochs=8,plot=False): 
        '''
        A function that trains a CNN given the model
        and the PATH of the data set.
        '''
        dump_dir = os.path.join(self.cnn_ds_path,'dump')
        #Process the Data
        image_gen = ImageDataGenerator(rescale=1./255)
        train_data_gen = image_gen.flow_from_directory(
                                    dump_dir,
                                    color_mode='grayscale',
                                    shuffle=True,
                                    target_size=(self.IMG_H, self.IMG_W),
                                    class_mode='input')
        
        self.autoencoder.fit(train_data_gen,
                            epochs=epochs,
                            batch_size=batch,
                            shuffle=True)

        self.autoencoder.save_weights(self.AE_save_path)

        
        
    def raw_featAutoEnc(self):
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

    def cluster_hdbscan(self,method='UMAP',n=3):
        '''
        Method that takes data points and cluster them using hdbscan. 
        args:
        returns:
        '''
        features = method+'-'+str(n)+'components-features.csv'
        data = pd.read_csv(os.path.join(self.cnn_ds_path,features))
        
        features = data.drop(columns=['Image Name','Unnamed: 0'])
        image_names = data.pop('Image Name')

        cluster = hdbscan.HDBSCAN(min_cluster_size=40,
                                min_samples=5,
                                cluster_selection_epsilon=0.1)
        cluster.fit(features.to_numpy())
        data['labels'] = cluster.labels_
        data['Image Names'] = image_names.to_numpy()

        print(data.head())
        data.to_csv(os.path.join(self.cnn_ds_path,method+'-'+str(n)+'D-clusters.csv'))

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        # Creating color map
        my_cmap = plt.get_cmap()

        x = data[method+' 0'].to_numpy()
        y = data[method+' 1'].to_numpy()
        z = data[method+' 2'].to_numpy()
        labels = data['labels'].to_numpy()

        sctt = ax.scatter(x,y,z,
                    alpha= 0.8,
                    c = labels,
                    cmap=my_cmap,
                    marker='^')
        fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
        plt.show()

    def createCNN_DS(self,file):
        '''
        Method that creates full data set for cnn training.
        args: None
        returns: None
        '''
        data_path = os.path.join(self.cnn_ds_path,'clusters')
        dump_path = os.path.join(self.cnn_ds_path,'dump')
        data = pd.read_csv(os.path.join(self.cnn_ds_path,file)).drop(columns=['Unnamed: 0','Unnamed: 0.1'])
        os.system('rm -rf '+str(os.path.join(data_path,'*')))
        print(data.head())

        # Made folder to seperate images
        paths = []
        indexes = [-1]
        indexes += list(range(int(max(data['labels']))+2))
        for i in indexes:
            name = os.path.join(data_path,str(i))
            os.mkdir(name)
            for j in range(len(data)):
                if data['labels'][j]==i:
                    shutil.copy(os.path.join(dump_path, data['Image Names'][j]), name)



