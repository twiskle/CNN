#!/usr/bin/python
# Created by: Pchu (Oct 2018)

'''
CnnSteps:

this class executes the CNN model based on the following methods:
1. set_input_data: input image data
2. Pre_Model_Steps: data preprocessing
3. create_baseline: create multi-layer training using Keras
4. Final_Model: creating the final CNN model
5. Result_Figure: display figures


Python 3.7.5
TensorFlow 2.0.0
Keras 2.2.4

12/16/19: updated changes for 2-input model (T1 and MM)
'''

##########################################################################


import os
from optparse import OptionParser
from os import sys,path,system
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import math

# model libraries
import tensorflow as tf
#from tensorflow import keras
#
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras import optimizers
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization

#print("tensorflow version: "+tf.__version__)


###use CPU over GPU

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
##########################################################################


class CnnSteps:

    dryRun = False


    def __init__(self, random_state, directories, x_Lsize, y_Lsize, z_Lsize):
        self.random_state = random_state
        self.directories = directories
        self.x_Lsize = x_Lsize
        self.y_Lsize = y_Lsize
        self.z_Lsize = z_Lsize


    def show_VRAM():
        ##show VRAM usage
        #bash line: watch nvidia-sm

        import tensorflow as tf
        import tensorflow.keras.backend.tensorflow_backend as K

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        K.set_session(sess)


    def execute(cmd):
        '''
        display each command in the terminal when being executed
        '''

        print("->"+cmd+"\n")
        if not dryRun:
            system(cmd)


    def pad_with(vector, pad_width, iaxis, kwargs):

        '''
        Add Zero-padding to the dataset
        '''

        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector

    def set_input_data(self):

        '''
        getting the input datasets
        '''

        X = []
        Y = []
        
        for i in range(len(self.directories)):
            for fileName in os.listdir(self.directories[i]):
                if fileName.endswith(".nii.gz"):
    #                print (fileName)
                    img = nib.load(path.join(self.directories[i],fileName))
                    
                    temp = np.array(img.get_data())
                    #Image Standardization: zero mean and 1 variance
                    temp = (temp - np.mean(temp))/np.std(temp)
                    if len(X)<1:
                        '''
                        if np.size(temp,0) < self.x_Lsize:
                            (x_L1,x_L2) = ( int(math.floor(float(
                            self.x_Lsize-np.size(temp,0))/2)), 
                            int(math.ceil(float(
                            self.x_Lsize-np.size(temp,0))/2))+1 )
                        else:
                            (x_L1,x_L2) = (0,1)
                        if np.size(temp,1) < y_Lsize:
                            (y_L1,y_L2) = ( int(math.floor(float(
                            self.y_Lsize-np.size(temp,1))/2)),  
                            int(math.ceil(float(
                            self.y_Lsize-np.size(temp,1))/2))+1 )
                        else:
                            (y_L1,y_L2) = (0,1)
                        if np.size(temp,2) < z_Lsize:
                            (z_L1,z_L2) = ( int(math.floor(float(
                            self.z_Lsize-np.size(temp,2))/2)), 
                            int(math.ceil(float(
                            self.z_Lsize-np.size(temp,2))/2))+1 )
                        else:
                            (z_L1,z_L2) = (0,1)
                        temp= np.pad(temp,((x_L1,x_L2),(y_L1,y_L2),
                            (z_L1,z_L2)),pad_with)
                        '''
                        X = np.expand_dims(temp, axis=0)
                    else:                       
                        '''
                        temp = np.pad(temp,((x_L1,x_L2),(y_L1,y_L2),
                            (z_L1,z_L2)),pad_with)
                        '''
                        temp = np.expand_dims(temp, axis=0)
                        X = np.append(X,temp,axis=0)
                    if "MS" in fileName:
                        Y = np.append(Y,np.array([1]))
                    elif "CTN" in fileName:
                        Y = np.append(Y,np.array([2]))


        print('Data shape : ', X.shape, Y.shape)


        # Find the unique numbers from the train labels
        classes = np.unique(Y)
        nClasses = len(classes)
        print('Total number kinds of output : ', nClasses)
        print('Output classes : ', classes)

        return ([X,Y])


    def Pre_Model_Steps(self):
        ##1. Set input data
        [X,Y] = self.set_input_data()
        X = np.array(X)

        ##2. encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        
        ##3. Reshape
        X = self.data_reshape(X,np.size(X,1),np.size(X,2),np.size(X,3))


        return [X,encoded_Y]

    def data_reshape(self,data_X,x,y,z):
        '''
        DATA Reshape
        '''
        #55,39,33
        #193,229,193
        #256,256,44

        data_X = data_X.reshape(-1, x,y,z, 1)
        print("reshaped to: ")
        print(data_X.shape)
        return data_X


    def create_baseline(self,num_classes,x,y,z):
        '''
        baseline model
        '''

        inp = Input(shape=(x,y,z,1))

        step = Conv3D(256, kernel_size=(3, 3, 3), strides=(1,1,1), 
            padding='same', activation='relu')(inp)
        step = MaxPooling3D((2, 2, 2),padding='same')(step)
        step = Conv3D(128, kernel_size=(3, 3, 3), strides=(1,1,1), 
            padding='same', activation='relu')(step)
        step = MaxPooling3D((2, 2, 2),padding='same')(step)
        step = Conv3D(16, kernel_size=(3, 3, 3), strides=(1,1,1), 
            padding='same', activation='relu')(step)
        step = MaxPooling3D((2, 2, 2),padding='same')(step)
        final_step = Flatten()(step)

        return inp,final_step


    def FinalModel(self,num_classes,set1_train_X, set2_train_X):
        #,set3_train_X,set4_train_X):
        '''
        Final CNN Model
        '''
        Input_1,branch1 = self.create_baseline(num_classes,np.size(
            set1_train_X,1),np.size(set1_train_X,2),np.size(set1_train_X,3))
        Input_2,branch2 = self.create_baseline(num_classes,np.size(set2_train_X,1),
            np.size(set2_train_X,2),np.size(set2_train_X,3))
        '''
        Input_3,branch3 = create_baseline(num_classes,np.size(set3_train_X,1),
            np.size(set3_train_X,2),np.size(set3_train_X,3))
        Input_4,branch4 = create_baseline(num_classes,np.size(set4_train_X,1),
            np.size(set4_train_X,2),np.size(set4_train_X,3))
        '''

        x1 = Model(inputs=Input_1,outputs=branch1)
        x2 = Model(inputs=Input_2,outputs=branch2)


        model_concat = concatenate([branch1,branch2])
        model_concat = Dense(128, activation='relu')(model_concat)
        model_concat = Dense(1, activation='sigmoid')(model_concat)

        model_combined = Model(inputs=[Input_1,Input_2],outputs=model_concat)
        model_combined.compile(optimizer='adam', 
            loss='binary_crossentropy',metrics=['accuracy'])
        '''
        model_combined.compile(loss='mean_squared_error', optimizer = RMSprop())
        model_combined.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy')
        '''
        #model_combined.summary()

        return model_combined


    def Result_Figure(self,train,saveImagePath,saveImageName):

        plt.figure(figsize=(10,10))
        '''
        plt.subplot(2,1,1)
        accuracy = train.history['acc']
        val_accuracy = train.history['val_acc']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        '''

        #plt.subplot(2,1,2)
        loss = train.history['loss']
        val_loss = train.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.legend()

        #plt.show()
        plt.savefig(os.path.join(saveImagePath,saveImageName))


    def train_model(model,x1train,x2train,x3train, y1train, 
        x1val,x2val,x3val, y1val):
        model.fit([x1train,x2train,x3train], y1train, batch_size=batch_size,
            epochs=epochs,verbose=1, validation_data=([x1val,x2val,x3val], 
            y1val))
        return model


