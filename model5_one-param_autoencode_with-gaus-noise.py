import os
from os import sys,path,system
import tensorflow as tf
from tensorflow import keras
import nibabel as nib

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import math

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("tensorflow version: "+tf.__version__)


###use CPU over GPU

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
##########################################################################


import time
time.sleep(30)




##show VRAM usage

#bash line: watch nvidia-sm
'''
import tensorflow as tf
import keras.backend.tensorflow_backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
'''

try:
    random_state = int(sys.argv[1])
except:
    print "Incorrect input(s). Please verify" 
    print "1st input: random state number"

try:
    step_num = int(sys.argv[2])
except:
    print "Incorrect input(s). Please verify" 
    print "2nd input: current step number"
    exit()



save_path="/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/Results"
model_suffix = "_DTI_1-param_epoch50_6fold_step%s" % (step_num)


dryRun = False

#getting the largest dimension of the images, and do zero padding on the rest of the image

x_Lsize = 256
y_Lsize = 256
z_Lsize = 193



'''
####1. Example dataset: fashion categorization
fashion_mnist = keras.datasets.fashion_mnist

#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''


def execute(cmd):
    print("->"+cmd+"\n")
    if not dryRun:
        system(cmd)


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def set_input_data(foldersPath):

    X = []
    Y = []
    
    for i in range(len(foldersPath)):
        for fileName in os.listdir(foldersPath[i]):
            if fileName.endswith(".nii.gz"):
#                print (fileName)
                img = nib.load(path.join(foldersPath[i],fileName))
                if len(X)<1:
                    temp = np.array(img.get_data())
                    '''
                    if np.size(temp,0) < x_Lsize:
                        (x_L1,x_L2) = ( int(math.floor(float(x_Lsize-np.size(temp,0))/2)),  int(math.ceil(float(x_Lsize-np.size(temp,0))/2))+1 )
                    else:
                        (x_L1,x_L2) = (0,1)
                    if np.size(temp,1) < y_Lsize:
                        (y_L1,y_L2) = ( int(math.floor(float(y_Lsize-np.size(temp,1))/2)),  int(math.ceil(float(y_Lsize-np.size(temp,1))/2))+1 )
                    else:
                        (y_L1,y_L2) = (0,1)
                    if np.size(temp,2) < z_Lsize:
                        (z_L1,z_L2) = ( int(math.floor(float(z_Lsize-np.size(temp,2))/2)),  int(math.ceil(float(z_Lsize-np.size(temp,2))/2))+1 )
                    else:
                        (z_L1,z_L2) = (0,1)
                    temp= np.pad(temp,((x_L1,x_L2),(y_L1,y_L2),(z_L1,z_L2)),pad_with)
                    '''
                    X = np.expand_dims(temp, axis=0)
                else:
                    temp = np.array(img.get_data())
                    '''
                    temp = np.pad(temp,((x_L1,x_L2),(y_L1,y_L2),(z_L1,z_L2)),pad_with)
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


#    print (Y)
#    raw_input("***pchu****: check output classes")

    return ([X,Y])

'''
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))
'''

####DATA Preprocessing

def data_preprocessing(data_X,x,y,z):

    #55,39,33
    #193,229,193
    #256,256,44

    data_X = data_X.reshape(-1, x,y,z, 1)
    print("reshaped to: ")
    print(data_X.shape)
    return data_X


##encode class values as integers
#encoder = LabelEncoder()
#encoder.fit(train_Y)
#encoded_Y = encoder.transform(train_Y)


# baseline model
def create_baseline(num_classes,x,y,z):

    from keras import Input
    from keras.models import Model
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv3D, MaxPooling3D, UpSampling3D
    from keras.layers import BatchNormalization
    from keras.layers import LeakyReLU
    from keras.layers import Activation
    from keras import optimizers

    inp = Input(shape=(x,y,z,1))

    step = Conv3D(128, kernel_size=(3, 3, 3), strides=(1,1,1), padding='same', activation='relu')(inp)
    step = MaxPooling3D((2, 2, 2),padding='same')(step)
    step = Conv3D(64, kernel_size=(3, 3, 3), strides=(1,1,1), padding='same', activation='relu')(step)
    step = MaxPooling3D((2, 2, 2),padding='same')(step)
    step = Conv3D(16, kernel_size=(3, 3, 3), strides=(1,1,1), padding='same', activation='relu')(step)
    step = MaxPooling3D((2, 2, 2),padding='same')(step)
    
    step = Conv3D(16, kernel_size=(3, 3, 3), strides=(1,1,1), padding='same', activation='relu')(step)
    step = UpSampling3D((2, 2, 2))(step)
    step = Conv3D(64, kernel_size=(3, 3, 3), strides=(1,1,1), padding='same', activation='relu')(step)
    step = UpSampling3D((2, 2, 2))(step)
    step = Conv3D(128, kernel_size=(3, 3, 3), strides=(1,1,1), padding='same', activation='relu')(step)
    step = UpSampling3D((2, 2, 2))(step)
    
    final_step = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(step)





#    step = Flatten()(step)
#    final_step = Dense(num_classes, activation='softmax')(step)

    
    #model.add(Dense(num_classes, activation='softmax')) # activation: use sigmoid for binary, else use softmax
    #model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    

    return inp,final_step


def FinalModel(num_classes,set1_train_X,set2_train_X,set3_train_X,set4_train_X):
    Input_1,branch1 = create_baseline(num_classes,np.size(set1_train_X,1),np.size(set1_train_X,2),np.size(set1_train_X,3))
#    Input_2,branch2 = create_baseline(num_classes,np.size(set2_train_X,1),np.size(set2_train_X,2),np.size(set2_train_X,3))
#    Input_3,branch3 = create_baseline(num_classes,np.size(set3_train_X,1),np.size(set3_train_X,2),np.size(set3_train_X,3))
#    Input_4,branch4 = create_baseline(num_classes,np.size(set4_train_X,1),np.size(set4_train_X,2),np.size(set4_train_X,3))

    from keras.models import Model
    from keras.layers import Dense
    from keras.layers import concatenate
    from keras.layers import BatchNormalization


#Autoencoder
    autoencoder = Model(Input_1,branch1)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')#,metrics=['accuracy'])
    #autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

    #autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    autoencoder.summary()

    return autoencoder


#1
#    merge = Dense(num_classes, activation='softmax')(branch1)
#    model = Model(inputs=Input_1,outputs=merge)


#2
#    merge = concatenate([branch1,branch2])
#    merge = Dense(256, activation='relu')(merge)
#    merge = BatchNormalization()(merge)
#    merge = Dense(256, activation='relu')(merge)
#    merge = BatchNormalization()(merge)
#    merge = Dense(256, activation='relu')(merge)
#    merge = BatchNormalization()(merge)
#    merge = Dense(num_classes, activation='softmax')(merge)
#    model = Model(inputs=[Input_1,Input_2],outputs=merge)


#4
#    merge = concatenate([branch1,branch2,branch3,branch4])
#    merge = Dense(256, activation='relu')(merge)
#    merge = BatchNormalization()(merge)
#    merge = Dense(256, activation='relu')(merge)
#    merge = BatchNormalization()(merge)
#    merge = Dense(256, activation='relu')(merge)
#    merge = BatchNormalization()(merge)
#    merge = Dense(num_classes, activation='softmax')(merge)
#    model = Model(inputs=[Input_1,Input_2,Input_3,Input_4],outputs=merge)

    '''    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    model.summary()

    return model
    '''


def Create_train_test_set(train_X,train_Y,test_X,test_Y):


    from keras.utils import to_categorical

    # Change the labels from categorical to one-hot encoding
    train_Y_one_hot = to_categorical(train_Y)
    test_Y_one_hot = to_categorical(test_Y)

    # Display the change for category label using one-hot encoding
    print('Original label:', train_Y[0])
    print('After conversion to one-hot:', train_Y_one_hot[0])



    from sklearn.model_selection import train_test_split
    train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=random_state)

    #print shape of train and test sets

    print ("train data shape: ", train_X.shape)
    print ("train label shape: ", train_label.shape)

    print ("train valid shape: ", valid_X.shape)
    print ("train valid label shape: ", valid_label.shape)


    return ([train_X,train_label],[valid_X,valid_label],test_Y_one_hot)


##Setting up Model

'''
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import optimizers

batch_size = 64  ## or 128 / 256 depending on memory size
epochs = 20
num_classes = 2

model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3),activation='relu',input_shape=(55,39,33,1),padding='same'))
#model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling3D((2, 2, 2),padding='same'))
model.add(Conv3D(64, (3, 3, 3), activation='relu',padding='same'))
#model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling3D(pool_size=(2, 2, 2),padding='same'))
model.add(Conv3D(128, (3, 3, 3), activation='relu',padding='same'))
#model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling3D(pool_size=(2, 2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))

## Compile the Model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])

model.summary()
'''

def Result_Figure(train,saveImagePath,saveImageName):

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


def train_model(model,x1train,x2train,x3train, y1train, x1val,x2val,x3val, y1val):
    model.fit([x1train,x2train,x3train], y1train, batch_size=batch_size,epochs=epochs,verbose=1, validation_data=([x1val,x2val,x3val], y1val))
    return model

########
########
########
########
###MAIN


import time
t0 = time.time()


##Dataset
'''
foldersPath1= ["/media/truecrypt1/Powell/deep_learning/MM_CTN_MSTN_5d/MSTN/betted_trunc", 
    "/media/truecrypt1/Powell/deep_learning/MM_CTN_MSTN_5d/CTN/betted_trunc"]
'''

##mni_space

#foldersPath1= ["/media/truecrypt1/Powell/deep_learning/T1_CTN_MSTN/MSTN/pons_masked_trunc", 
#    "/media/truecrypt1/Powell/deep_learning/T1_CTN_MSTN/CTN/pons_masked_trunc"]


foldersPath1= ["/media/truecrypt1/Powell/deep_learning/T1_CTN_MSTN/MSTN/brainstem_masked_trunc", 
    "/media/truecrypt1/Powell/deep_learning/T1_CTN_MSTN/CTN/brainstem_masked_trunc"]



##subject_space
foldersPath4= ["/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/MSTN/FA/betted", 
    "/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/CTN/FA/betted"]

foldersPath2= ["/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/MSTN/MD/betted", 
    "/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/CTN/MD/betted"]

foldersPath3= ["/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/MSTN/AD/betted", 
    "/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/CTN/AD/betted"]


def Pre_Model_Steps(foldersPath):
    ##1. Set input data
    [X,Y] = set_input_data(foldersPath)
    X=np.array(X)
    
    ##2. Preprocessing
    X=data_preprocessing(X,np.size(X,1),np.size(X,2),np.size(X,3))

    ##3. encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)


    return [X,encoded_Y]


[X1,encoded_Y1] = Pre_Model_Steps(foldersPath1)
[X2,encoded_Y2] = Pre_Model_Steps(foldersPath2)
[X3,encoded_Y3] = Pre_Model_Steps(foldersPath3)
[X4,encoded_Y4] = Pre_Model_Steps(foldersPath4)

##4. create model
batch_size = 4  ## or 128 / 256 depending on memory size
epochs = 30
num_classes = 2
n_splits = 6

'''
model = FinalModel(num_classes,set1_train_X,set2_train_X,set3_train_X)


##5. Train the model
#train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
train = model.fit([set1_train_X,set2_train_X,set3_train_X], set1_train_label, batch_size=batch_size,epochs=epochs,verbose=1, validation_data=([set1_valid_X,set2_valid_X,set3_valid_X], set1_valid_label))


##6. Evaluate on Test Set
test_eval = model.evaluate([set1_test_X,set2_test_X,set3_test_X], set1_test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

##7. Result Figures
save_path="/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/Results"
saveImageName="results_DTI_FA_RD_AD_betted_epoch10.png"
Result_Figure(train,save_path,saveImageName)

##8. Saving model
model.save(path.join(save_path,"test_FA_RD_AD_epoch10.h5py"))
'''


# evaluate model with standardized dataset
#estimator = KerasClassifier(build_fn=FinalModel(num_classes,X1,X2), epochs=epochs, batch_size=batch_size, verbose=0)
#estimator = KerasClassifier(build_fn=FinalModel(num_classes,X1,X2,X3,X4), epochs=epochs, batch_size=batch_size, verbose=0)

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

#results = cross_val_score(estimator, [X1,X2,X3], [encoded_Y1,encoded_Y2,encoded_Y3], cv=kfold)
#print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


##9. Clear swap memory before next iteration
#cmd = ('sudo swapoff -a && sudo swapon -a').format(**dict(locals(), **globals()))
#execute(cmd)


# Loop through the indices the split() method returns
(index, (train_indices,val_indices)) = list(enumerate(kfold.split(X1, encoded_Y1)))[step_num]
print "Training on fold " + str(step_num+1) + "/" + str(kfold.n_splits) +"..."
# Generate batches from indices
x1train, x1val = X1[train_indices], X1[val_indices]
#x2train, x2val = X2[train_indices], X2[val_indices]
#x3train, x3val = X3[train_indices], X3[val_indices]
#x4train, x4val = X4[train_indices], X4[val_indices]

x1train = x1train.astype('float32') / np.max(x1train)
x1val = x1val.astype('float32') / np.max(x1val)


y1train, y1val = encoded_Y1[train_indices], encoded_Y1[val_indices]


# Clear model, and create it
model = None

#    from keras import backend as K
#    tf_session = K.get_session()

model = FinalModel(num_classes,X1,X2,X3,X4)

X2 = None       ## trying out 1 parameter first 
X3 = None
X4 = None


'''
from Xlib import display, X
from PIL import Image #PIL

W,H = 1920,1080
dsp = display.Display()
root = dsp.screen().root
raw = root.get_image(0, 0, W,H, X.ZPixmap, 0xffffffff)
image = Image.frombytes("RGB", (W, H), raw.data, "raw", "BGRX")
image.save(path.join(save_path,"memory"+model_suffix+"_step"+str(step_num)+"_beginning.jpg"))
'''


##Add Noise (Gaussian)
noise_factor = 0.5
x1train_noisy = x1train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x1train.shape) 
x1val_noisy = x1val + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x1val.shape) 

x1train_noisy = np.clip(x1train_noisy, 0., 1.)
x1val_noisy = np.clip(x1val_noisy, 0., 1.)

# Debug message I guess
print "Training new iteration on " + str(x1train.shape[0]) + " training samples, " + str(x1val.shape[0]) + " validation samples, this may be a while..."
print "Training indices: " + str(train_indices)
print "Validation indices: " + str(val_indices)
print "batch size: %s, epochs = %s" % (batch_size, epochs)


#train = model.fit(x1train, x1train, batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(x1val, x1val))

train = model.fit(x1train_noisy, x1train, batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(x1val_noisy, x1val))



##7. Result and Figures

if step_num < 1:
    results_txt = open(path.join(save_path,'results'+model_suffix+'.txt'),'w')
else:
    results_txt = open(path.join(save_path,'results'+model_suffix+'.txt'),'a')

import re
results_txt.write("Stratified kfold, with number of folds: "+str(kfold.n_splits)+"\n")
results_txt.write("Step "+str(step_num)+"\n")
results_txt.write("batch size: %s, epochs = %s \n" % (batch_size, epochs))
results_txt.write("Training indices: \n")
results_txt.writelines(str(train_indices)+"\n")
results_txt.write("Validation indices: \n")
results_txt.writelines(str(val_indices)+"\n")
results_txt.write(re.sub("[|]|'|,","",str(train.history.keys()))+"\n")
results_txt.write(re.sub("[|]|'|,","",str(train.history.values()))+"\n")


saveImageName="figure"+model_suffix+".png"
Result_Figure(train,save_path,saveImageName)


##8. Saving model
model.save(path.join(save_path,"model"+model_suffix+".h5py"))


##9. Clear swap memory before next iteration
#    cmd = ('sudo swapoff -a && sudo swapon -a').format(**dict(locals(), **globals()))
#    execute(cmd)


#from keras import backend as K
#model = K.clear_session()


import time
time.sleep(30)
'''
W,H = 1920,1080
dsp = display.Display()
root = dsp.screen().root
raw = root.get_image(0, 0, W,H, X.ZPixmap, 0xffffffff)
image = Image.frombytes("RGB", (W, H), raw.data, "raw", "BGRX")
#image.save(path.join(save_path,"memory"+model_suffix+"_step"+str(step_num)+"_end.jpg"))
'''

'''
# summary history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
'''


#history = train_model(model, x1train, x2train, x3train, y1train, x1val, x2val, x3val, y1val)
#accuracy_history = history.history['acc']
#val_accuracy_history = history.history['val_acc']
#print "Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(val_accuracy_history[-1])

results_txt.write("Time from start to now is (min): "+str((time.time()-t0)/60.0)+"\n")
results_txt.close()


print ("DONE!")
t1 = time.time()
total_time = t1-t0
print ("total time: ",total_time/60.0,"   min")