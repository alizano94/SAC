import os
import pandas as pd
import numpy as np
from random import seed, randint
import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing import image

class Helpers():
    def __init__(self):
        self.IMG_H=212
        self.IMG_W=212
        self.chan=1

        self.cnn_results_path = '/home/lizano/Documents/SAC/results/cnn'
        self.cnn_ds_path = '/home/lizano/Documents/SAC/data/raw/cnn'
        self.cnn_weights_path = '/home/lizano/Documents/SAC/models/cnn'
        self.cnn_preprocess_data_path = '/home/lizano/Documents/SAC/data/preprocessed/cnn'

        self.snn_results_path = '/home/lizano/Documents/SAC/results/snn'
        self.snn_ds_path = '/home/lizano/Documents/SAC/data/raw/snn'
        self.snn_weights_path = '/home/lizano/Documents/SAC/models/snn'
        self.snn_preprocess_data_path = '/home/lizano/Documents/SAC/data/preprocessed/snn'

        self.contorl_policies = '/home/lizano/Documents/SAC/models/control'

        self.k = len(os.listdir(os.path.join(self.cnn_preprocess_data_path,'train')))

    def preProcessImg(self,img_path):
        '''
        A function that preprocess an image to fit 
        the CNN input.
        args:
            -img_path: path to get image
            -IMG_H: image height
            -IMG_W: image width
        Returns:
            -numpy object containing:
                (dum,img_H,img_W,Chanell)
        '''
        #Load image as GS with st size
        img = image.load_img(img_path,color_mode='grayscale',target_size=(self.IMG_H, self.IMG_W))
        #save image to array (H,W,C)
        img_array = image.img_to_array(img)
        
        #Create a batch of images
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    def windowResampling(slef,data,sampling_ts,window,memory):
        '''
        Receives data in a dataframe and returns data frame 
        with resampled data using slinding window method
        '''
        standard = ['Time','C6_avg','psi6','V']
        columns = []+standard
        for i in range(memory+1):
            name = 'S'+str(-memory+i)
            columns += [name]

        out_df = pd.DataFrame(columns=columns)
        new_size = int(len(data) - memory*window/sampling_ts)
        #refactor this to use all the data set competting with augmented data

        for index, rows in data.iterrows():
            row = {}
            if index < new_size:
                for name in standard:
                    i = int(index+(memory-1)*window/sampling_ts)
                    row[name] = data.at[i,name]
                for m in range(memory+1):
                    name = 'S'+str(-memory+m)
                    i = int(index+m*window/sampling_ts)
                    row[name] = data.at[i,'S_cnn']
                #print(row)
                out_df = out_df.append(row,ignore_index=True)
        

        return out_df

    def df2dict(self,df,dtype=float):
        '''
        Takes a df and returns a dict of tensors
        '''
        out_dict = {name: np.array(value,dtype=dtype)
                    for name, value in df.items()}

        return out_dict

    def onehotencoded(self,df,dtype=float):
        '''
        Transforms array with out state into one hot encoded vector
        '''
        array = np.array(df['S0'],dtype=dtype)
        onehotencoded_array = np.zeros((len(array),self.k),dtype=int)
        for i in range(len(array)):
            index = int(array[i])
            onehotencoded_array[i][index] = 1

        return onehotencoded_array

    def balanceData(self,data,method='drop'):
        '''
        Resamples the data to ensure theres no BIAS on 
        ouput state dsitribution.
        '''
        seed(1)

        hist = self.getHist(data)

        min_hist = min(hist)
        max_hist = max(hist)

        if method == 'drop':
            while max(hist) != min_hist:
                index = randint(0,len(data)-1)
                hist_index = int(data['S0'][index])
                if hist[hist_index] > min_hist:
                    data.drop(index=index, inplace=True)
                hist = self.getHist(data)
                print(hist)
                data.reset_index(inplace=True)
                data.drop(columns=['index'],inplace=True)
        else:
            while min(hist) != max_hist:
                index = randint(0,len(data)-1)
                hist_index = int(data['S0'][index])
                if hist[hist_index] < max_hist:
                    #add random value to data set
                    data.loc[len(data.index)] = data.iloc[index]
                hist = self.getHist(data)
                print(hist)
                data.reset_index(inplace=True)
                data.drop(columns=['index'],inplace=True)

        return data

    def getHist(self,data):
        '''
        create histogram from labels data
        '''
        hist = []
        for i in range(self.k):
            hist.append(0)

        for _, rows in data.iterrows():
            hist[int(rows['S0'])] += 1
        return hist

    def stateEncoder(self,k,state):
        '''
        Method that encondes a given state into 
        a number
        '''
        s = 0
        m = len(state)
        for i in range(m):
            j = m - i -1
            s += state[j]*k**i
        return s

    def stateDecoder(self,k,state,m):
        '''
        Method that decodes stae from number to 
        input vector.
        '''
        done = False
        out = []
        q,r = 0,0
        q = state
        while not done:
            new_q = q // k
            #print(new_q)
            r = q % k
            q = new_q
            out.append(r)
            if new_q == 0:
                done = True
        while len(out) < m:
            out.append(0)

        #print(out)
        out = out[::-1]
        return out

    def plot3Dscatter(self,method = 'tSNE'):
        '''
        Method that plots a 3D scatter plot from dataframe.
        args:
            -data: DataFrame containing the points
        returns: None
        '''
        data = pd.read_csv(os.path.join(self.cnn_ds_path,method+'-3components-features.csv'))
        x = data[method+' 0'].tolist()
        y = data[method+' 1'].tolist()
        z = data[method+' 2'].tolist()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x,y,z)
        ax.set_xlabel(method+' 0')
        ax.set_ylabel(method+' 1')
        ax.set_zlabel(method+' 2')
        plt.show()


        
