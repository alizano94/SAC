import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
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

class Autoencoder(Helpers):
    def __init__(self,*args,**kwargs):
        super(Autoencoder, self).__init__(*args, **kwargs)
        self.AE_save_path = os.path.join(self.cnn_weights_path,'AE.h5')

    def createAE(self,latent_dim=1000,summary=False):
        '''
        Creates autoencoder with specified latent space.
        args:
            -latent_dim: size of the latent space
            -summary: False by default, set to True
                        to get sumary of the model.
        returns: None
        '''
        self.latent_dim = latent_dim

        inputs = tf.keras.Input(shape=(self.IMG_H,self.IMG_W,self.chan), name='Image object input')
        self.encoder = layers.Flatten()(inputs)
        self.encoder = layers.Dense(self.latent_dim, activation='relu')(self.encoder)
        
        self.decoder = layers.Dense(self.IMG_H*self.IMG_W, activation='sigmoid')(self.encoder)
        self.decoder = layers.Reshape((self.IMG_H, self.IMG_W))(self.decoder)
        
        self.autoencoder = Model(inputs=inputs,outputs=self.decoder)
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
        
        history = self.autoencoder.fit(train_data_gen,
                            batch_size=batch,
                            epochs=epochs,
                            shuffle=True,
                            callbacks = [tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=0.01,
                            patience=7)])
        
        if plot:
            #Plot Accuracy, change this to matplotlib
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history.epoch,
                                y=history.history['val_loss'],
                                mode='lines+markers',
                                name='Training loss'))
            fig.update_layout(title='Reconstruction Loss',
                        xaxis=dict(title='Epoch'),
                        yaxis=dict(title='Loss'))
            fig.show()

        self.autoencoder.save_weights(self.AE_save_path)

    def loadAE(self,path):
        '''
        Functions that loads weight for the model
        args:
			-path: path from which to load weights
		'''
        if path == None:
            path = self.AE_save_path
        #Load model wieghts
        self.autoencoder.load_weights(path)
        print("Loaded model from disk")

class Autoencoder_Test(Autoencoder):
    def __init__(self,*args,**kwargs):
        super(Autoencoder_Test,self).__init__(*args,**kwargs)
        #Add Testing for autoencoder learning


class IMG_Clustering(Autoencoder):
    def __init__(self, *args, **kwargs):
        super(IMG_Clustering, self).__init__(*args, **kwargs)

    def raw_featInception(self,path=None,out_name='raw_features.csv',data_set='train'):
        '''
        Method that takes a dump of images and extracts their features
        using the InceptionV3 model.
        args:None
        returns:
            -data: DataFrame containing the raw features.
        '''
        if not path:
            path = os.path.join(self.cnn_ds_path,'unclassified_raw_data',data_set)
        model = InceptionV3(weights='imagenet', include_top=False)
        raw_features = []
        img_name = []
        img_path = os.listdir(path)
        for i in tqdm(img_path):
            fname=path+'/'+i
            img=image.load_img(fname,target_size=(224,224))
            x = img_to_array(img)
            x=np.expand_dims(x,axis=0)
            x=preprocess_input(x)
            feat=model.predict(x)
            feat=feat.flatten()
            raw_features.append(feat)
            img_name.append(i)
        columns_names = ['Image Names']
        columns_feat = []
        for i in range(len(raw_features[0])):
            header = 'raw '+str(i)
            columns_feat.append(header)
        img_name,raw_features = np.row_stack(img_name),np.row_stack(raw_features)
        img_name,raw_features = pd.DataFrame(img_name,columns=columns_names), pd.DataFrame(raw_features,columns=columns_feat)
        data = pd.concat([img_name,raw_features],axis=1,join='inner')
        print(data.head())

        data.to_csv(os.path.join(self.cnn_ds_path,'unclassified_raw_data',out_name))

    def tSNE(self,n,load_file='raw_features.csv',out_file='tsne_features.csv'):
        '''
        Performs tSNE reduction on the raw features for the images.
        args:
            -n: number of componets to reduce to
        return: None
        '''
        load_file = os.path.join(self.cnn_ds_path,'unclassified_raw_data',load_file)
        if not os.path.exists(load_file):
            print('File not found extracting features using Inception Model.')
            self.raw_featInception()
            load_file = os.path.join(self.cnn_ds_path,'unclassified_raw_data','raw_features.csv')

        data = pd.read_csv(load_file,index_col=0)
        raw_features = data.drop(columns=['Image Names'])
        image_names = data.pop('Image Names')

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
        data.to_csv(os.path.join(self.cnn_ds_path,'unclassified_raw_data',out_file))

    def umap(self,n,load_file='raw_features.csv',out_file='umap_features.csv'):
        '''
        Performs tSNE reduction on the raw features for the images.
        args:
            -n: number of componets to reduce to
        return: None
        '''
        
        load_file = os.path.join(self.cnn_ds_path,'unclassified_raw_data',load_file)
        if not os.path.exists(load_file):
            print('File not found extracting features using Inception Model.')
            self.raw_featInception()
            load_file = os.path.join(self.cnn_ds_path,'unclassified_raw_data','raw_features.csv')

        data = pd.read_csv(load_file,index_col=0)
        raw_features = data.drop(columns=['Image Names'])
        image_names = data.pop('Image Names')

        raw_features.to_numpy()
        columns = []
        for i in range(n):
            header = 'UMAP '+str(i)
            columns.append(header)
        mapper = UMAP(n_components=n)
        features = mapper.fit_transform(raw_features)
        features = pd.DataFrame(features,columns=columns)
        data = pd.concat([image_names,features],axis=1,join='inner')
        data.to_csv(os.path.join(self.cnn_ds_path,'unclassified_raw_data',out_file))

    def cluster_hdbscan(self,plot=False,mcs=40,ms=5,eps=0.1,
                        metric='euclidean',
                        load_file='umap_features.csv',
                        out_file='hdbscan_clusters.csv'):
        '''
        Method that takes data points and cluster them using hdbscan. 
        args:
        returns:
        '''
        load_file = os.path.join(self.cnn_ds_path,'unclassified_raw_data',load_file)
        if not os.path.exists(load_file):
            print('File not found creating file using UMAP')
            self.umap(n=3)
            load_file = os.path.join(self.cnn_ds_path,'unclassified_raw_data','umap_features.csv')
        data = pd.read_csv(load_file,index_col=0)
        
        features = data.drop(columns=['Image Names'])
        image_names = data.pop('Image Names')

        cluster = hdbscan.HDBSCAN(min_cluster_size=mcs,
                                min_samples=ms,
                                cluster_selection_epsilon=eps,
                                metric=metric)
        cluster.fit(features.to_numpy())
        data['labels'] = cluster.labels_
        data['Image Names'] = image_names.to_numpy()

        #print(data.head())
        data.to_csv(os.path.join(self.cnn_ds_path,'unclassified_raw_data',out_file))


class CNN_Asistance(IMG_Clustering):
    def __init__(self,*args,**kwargs):
        super(CNN_Asistance,self).__init__(*args,**kwargs)
    
    def createCNN_DS(self,load_file='hdbscan_clusters.csv',delete=True):
        '''
        Method that creates full data set for cnn training.
        args: None
        returns: None
        '''
        load_file = os.path.join(self.cnn_ds_path,'unclassified_raw_data',load_file)
        if not os.path.exists(load_file):
            print('File not found creating file using HDBSCAN')
            self.cluster_hdbscan()
            load_file = os.path.join(self.cnn_ds_path,'unclassified_raw_data','hdbscan_clusters.csv')
        data = pd.read_csv(load_file,index_col=0)
        if delete:
            os.system('rm -rf '+str(os.path.join(self.cnn_ds_path,'clusters','*')))

        # Made folder to seperate images
        indexes = data['labels'].unique()
        for i in indexes:
            name = os.path.join(self.cnn_ds_path,'clusters',str(i))
            os.mkdir(name)
            for j in range(len(data)):
                if data['labels'][j]==i:
                    shutil.copy(os.path.join(self.cnn_ds_path,'unclassified_raw_data',
                                            'train',
                                            data['Image Names'][j])
                                ,name)

    def createCNN_spits(self,validation_split=0.2,testing_split=0.2):
        '''
        Function that creates the testing training and validation 
        splits for the CNN further trining. 
        args:
            -validation_split: fraction of total data set to take as validation.
            -testing_split: fraction of total data set to take as testing.
        return:
        '''
        import math
        
        #backup the clusters directory to a tempoaray one
        clusters_path = os.path.join(self.cnn_ds_path,'clusters')
        work_path = os.path.join(self.cnn_ds_path,'temp')
        os.system("rm -rf "+os.path.join(self.cnn_preprocess_data_path,"*"))        
        destinations = ["train","test","validation"]
        for i in range(len(destinations)):
            destinations[i] = os.path.join(self.cnn_preprocess_data_path,destinations[i])
            os.system("mkdir "+destinations[i])
        os.system("cp -r "+clusters_path+" "+work_path)

        size = 0
        for cluster in os.listdir(work_path):
            print(cluster)
            if cluster != "-1":
                path = os.path.join(work_path,cluster)
                print(path)
                number_files = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
                training_size = str(math.trunc(number_files/(1+validation_split+testing_split)))
                os.system("mkdir "+os.path.join(destinations[0],cluster))
                os.system("shuf -n"+training_size+" -e "+path+"/*.png | xargs -i mv {} "+os.path.join(destinations[0],cluster))
                testing_size = str(math.trunc(testing_split*number_files))
                os.system("mkdir "+os.path.join(destinations[1],cluster))
                os.system("shuf -n"+testing_size+" -e "+path+"/*.png | xargs -i mv {} "+os.path.join(destinations[1],cluster))
                validation_size = str(math.trunc(validation_split*number_files))
                os.system("mkdir "+os.path.join(destinations[2],cluster))
                os.system("shuf -n"+validation_size+" -e "+path+"/*.png | xargs -i mv {} "+os.path.join(destinations[2],cluster))
                size += number_files
        os.system("rm -rf "+work_path)
                

class Clustering_Test(IMG_Clustering):
    def __init__(self,*args,**kwargs):
        super(Clustering_Test,self).__init__(*args,**kwargs)
        #Add Testing for Clsutering


    def pca_reconstruction_error(self,n=2,n_samples=100,raw_feat_file='validation_raw_features.csv'):
        '''
        Function that takes projected features, calculates its mapping into
        the latent space and then gets the inverse transformation of it. 
        Finally calculates the reconstruction error.
        args:
        return:
        '''
        raw_feat_path = os.path.join(self.cnn_ds_path,'unclassified_raw_data',raw_feat_file)

        raw_features = pd.read_csv(raw_feat_path,index_col=0)
        raw_features.drop(columns=['Image Names'],inplace=True)
        raw_features = raw_features.to_numpy()
        raw_features = raw_features[:,:n_samples]

        mapper = PCA(n_components=n)
        print('Projecting features')
        projected_features = mapper.fit_transform(raw_features)
        print('Retrieving features')
        inv_features = mapper.inverse_transform(projected_features)
        print('Calculating RMSE')
        return np.sqrt(np.mean((inv_features-raw_features)**2))

    def umap_reconstruction_error(self,n=2,n_samples=100,raw_feat_file='validation_raw_features.csv'):
        '''
        Function that takes projected features, calculates its mapping into
        the latent space and then gets the inverse transformation of it. 
        Finally calculates the reconstruction error.
        args:
        return:
        '''
        raw_feat_path = os.path.join(self.cnn_ds_path,'unclassified_raw_data',raw_feat_file)

        raw_features = pd.read_csv(raw_feat_path,index_col=0)
        raw_features.drop(columns=['Image Names'],inplace=True)
        raw_features = raw_features.to_numpy()
        raw_features = raw_features[:,:n_samples]

        mapper = UMAP(n_components=n)
        print('Projecting features')
        projected_features = mapper.fit_transform(raw_features)
        print('Retrieving features')
        inv_features = mapper.inverse_transform(projected_features)
        print('Calculating RMSE')
        return np.sqrt(np.mean((inv_features-raw_features)**2))
        
