import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.dataPrep import CNN_Asistance

class CNN(CNN_Asistance):
    def __init__(self, *args, **kwargs):
        super(CNN, self).__init__(*args, **kwargs)
        self.save_path = os.path.join(self.cnn_weights_path,'CNN-'+str(self.k)+'states.h5')
        
    def createCNN(self,summary=False):
        '''
        function that creates and compile the CNN
        args:
        	-IMG_H:Images height (def 212px)
        	-IMH_W:Images width (def 212px)
            -chan: color channels (def 1 i.e. grayscale)
        '''
		
        # Model Creation
        self.cnn_model = Sequential()
        self.cnn_model.add(Conv2D(16, 3, padding='same', activation='relu',
			input_shape=(self.IMG_H, self.IMG_W, self.chan)))
        self.cnn_model.add(MaxPooling2D())
        self.cnn_model.add(Dropout(0.2))
        self.cnn_model.add(Conv2D(32, 3, padding='same', activation='relu'))
        self.cnn_model.add(MaxPooling2D())
        self.cnn_model.add(Conv2D(64, 3, padding='same', activation='relu'))
        self.cnn_model.add(MaxPooling2D())
        self.cnn_model.add(Dropout(0.2))
        self.cnn_model.add(Conv2D(128, 3, padding='same', activation='relu'))
        self.cnn_model.add(MaxPooling2D())
        self.cnn_model.add(Dropout(0.2))
        self.cnn_model.add(Conv2D(256, 3, padding='same', activation='relu'))
        self.cnn_model.add(MaxPooling2D())
        self.cnn_model.add(Dropout(0.2))
        self.cnn_model.add(Flatten())
        self.cnn_model.add(Dense(512, activation='relu'))
        self.cnn_model.add(Dense(self.k, activation='softmax'))

		#Compile the model
        self.cnn_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
        if summary:
            self.cnn_model.summary()
            tf.keras.utils.plot_model(
				model = self.cnn_model,
				rankdir="TB",
				dpi=72,
				show_shapes=True
				)
    
    def trainCNN(self,batch=32,epochs=8,plot=False): 
        '''
        A function that trains a CNN given the model
        and the PATH of the data set.
        '''
        train_dir = os.path.join(self.cnn_preprocess_data_path,'train')
        test_dir = os.path.join(self.cnn_preprocess_data_path,'test')

        #Process the Data
        image_gen = ImageDataGenerator(rescale=1./255)
        train_data_gen = image_gen.flow_from_directory(
                                    batch_size=batch,
                                    directory=train_dir,
                                    color_mode='grayscale',
                                    shuffle=True,
                                    target_size=(self.IMG_H, self.IMG_W),
                                    class_mode='categorical')

        test_data_gen = image_gen.flow_from_directory(
                                    batch_size=batch,
                                    directory=test_dir,
                                    color_mode='grayscale',
                                    target_size=(self.IMG_H, self.IMG_W),
                                    class_mode='categorical')
        history = self.cnn_model.fit(
                            train_data_gen,
                            steps_per_epoch=train_data_gen.n // train_data_gen.batch_size,
                            epochs=epochs,
                            validation_data=test_data_gen,
                            validation_steps=test_data_gen.n // test_data_gen.batch_size,
                            callbacks = [tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=0.01,
                            patience=7)])
        
        self.cnn_model.save_weights(self.save_path)
        
        if plot:
            #Plot Accuracy, change this to matplotlib
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history.epoch,
                                y=history.history['accuracy'],
                                mode='lines+markers',
                                name='Training accuracy'))
            fig.add_trace(go.Scatter(x=history.epoch,
                                y=history.history['val_accuracy'],
                                mode='lines+markers',
                                name='Validation accuracy'))
            fig.update_layout(title='Accuracy',
                        xaxis=dict(title='Epoch'),
                        yaxis=dict(title='Percentage'))
            fig.show()

    def loadCNN(self,path):
        '''
        Functions that loads weight for the model
        args:
			-path: path from which to load weights
		'''
        if path == None:
            path = self.save_path
        #Load model wieghts
        self.cnn_model.load_weights(path)
        print("Loaded model from disk")

    def runCNN(self,img_path):
        '''
        Method that receives image and returns label of image.
        args:
            -img_path: path to image to be classified. 
        returns:
            -label of the classified image.
            -probabilities of the classified image.
        '''
        img_batch = self.preProcessImg(img_path)
        prediction = self.cnn_model.predict(img_batch)
        #cat_index = np.argmax(prediction[0])
        
        return np.argmax(prediction[0]), prediction

class CNN_Testing(CNN):
    def __init__(self, *args, **kwargs):
        super(CNN_Testing, self).__init__(*args, **kwargs)
    
    def testCNN(self,path):
        '''
        Function that test CNN performance by callculating the 
        confusion matrix using the data in the specified path.
        args:
            -path: the path to the data sst to test. Set path = None
            to test using training/testing data set. 
        returns;
            -confution matrix: confusion matirx for the selected data.
            -missclassified data: csv containing the missclassified images.  
        '''
        if path == None:
            path = os.path.join(self.cnn_preprocess_data_path,'validation')

        Conf_Mat = np.zeros([self.k,self.k])
        error_log = pd.DataFrame(columns=['Image Name','Path','True Label','Predicted Label'])
        for cluster in os.listdir(path):
            for image in os.listdir(os.path.join(path,cluster)):
                pred_label, _ = self.runCNN(os.path.join(path,cluster,image))
                Conf_Mat[int(cluster),int(pred_label)] += 1
        print(Conf_Mat)
        self.plot_heatmap(Conf_Mat)
        



class SNN_Asistance(CNN):
    def __init__(self, *args, **kwargs):
        super(SNN_Asistance, self).__init__(*args, **kwargs)


    def preProcessSNNDS(self):
        '''
        Method that takes op1.txt and plots and creates 
        csv containing information about time, order params,
        real state, cnnn state and voltage.
        '''
        sep = '","'
        for v_dir in os.listdir(self.snn_ds_path):
            v_path = os.path.join(self.snn_ds_path,v_dir)
            if os.path.isdir(v_path):
                for step_dir in os.listdir(v_path):
                    step_path = os.path.join(v_path,step_dir)
                    if os.path.isdir(step_path):
                        for t_dir in os.listdir(step_path):
                            t_path = os.path.join(step_path,t_dir)
                            if os.path.isdir(t_path):
                                op_path = os.path.join(t_path,'op1.txt')
                                if os.path.exists(op_path):
                                    os.chdir(t_path)
                                    csv_name = v_dir+'-'+step_dir+'-'+t_dir+'.csv'
                                    os.system("awk '{print $1,"
                                        +sep+",$2,"
                                        +sep+",$3,"
                                        +sep+",$4,"
                                        +sep+",$5,"
                                        +sep+",$6,"
                                        +sep+",$7}' op1.txt > test.txt")
                                    data = pd.read_csv('test.txt', header=None)
                                    data.columns = ['Time','C6_avg','rgmean','psi6','RC','V','lambda']
                                    data = data.drop(labels=['RC','lambda','rgmean'],axis=1)
                                    states = pd.DataFrame(columns = ['S_cnn', 'S_param'])
                                    for i in range(0,len(data.index)):
                                        file_name = t_path+'/plots/'+v_dir+'-'+t_dir+'-'+str(i)+'step'+step_dir+'.png'
                                        s_cnn, _ = self.runCNN(file_name)
                                        c6 = data.iloc[i]['C6_avg']
                                        psi6 = data.iloc[i]['psi6']
                                        if c6 <= 4.0:
                                            s_real = 0
                                        elif c6 > 4.0 and psi6 < 0.99:
                                            s_real = 1
                                        else:
                                            s_real = 2
                                        states_dict = {'S_cnn':s_cnn,'S_param':s_real}
                                        states = states.append(states_dict,ignore_index=True)
                                    data = pd.concat([data,states], axis=1)
                                    print(data)
                                    data.to_csv(csv_name,index=False)
                                    os.system('rm -rf test.txt')
                                    os.chdir(self.snn_ds_path)

