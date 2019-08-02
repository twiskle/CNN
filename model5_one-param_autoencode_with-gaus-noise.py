import os
from optparse import OptionParser
from os import sys,path,system
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import math

# model libraries
import tensorflow as tf
from tensorflow import keras
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


# inputs and paths

dryRun = False

#getting the largest dimension of the images, and do zero padding on the rest of the image

x_Lsize = 256
y_Lsize = 256
z_Lsize = 193


save_path="/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/Results"
model_suffix = "_DTI_1-param_epoch50_6fold_step%s" % (step_num)

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
foldersPath2= ["/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/MSTN/FA/betted", 
    "/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/CTN/FA/betted"]

foldersPath3= ["/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/MSTN/MD/betted", 
    "/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/CTN/MD/betted"]

foldersPath4= ["/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/MSTN/AD/betted", 
    "/media/truecrypt1/Powell/deep_learning/DTI_CTN_MSTN/CTN/AD/betted"]






def show_VRAM:
    ##show VRAM usage
    #bash line: watch nvidia-sm

    import tensorflow as tf
    import keras.backend.tensorflow_backend as K

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)


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

    return ([X,Y])


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

def data_preprocessing(data_X,x,y,z):

    ####DATA Preprocessing

    #55,39,33
    #193,229,193
    #256,256,44

    data_X = data_X.reshape(-1, x,y,z, 1)
    print("reshaped to: ")
    print(data_X.shape)
    return data_X


def create_baseline(num_classes,x,y,z):
    # baseline model

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


############
############
##  MAIN  ##
############
############

def CnnModel(options,args):
    random_state = args[0]
    step_num = args[1]

    import time
    t0 = time.time()


    [X1,encoded_Y1] = Pre_Model_Steps(foldersPath1)
    [X2,encoded_Y2] = Pre_Model_Steps(foldersPath2)
    [X3,encoded_Y3] = Pre_Model_Steps(foldersPath3)
    [X4,encoded_Y4] = Pre_Model_Steps(foldersPath4)

    ##4. create model
    batch_size = 4  ## or 128 / 256 depending on memory size
    epochs = 30
    num_classes = 2
    n_splits = 6


    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


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
    cmd = ('sudo swapoff -a && sudo swapon -a').format(**dict(locals(), **globals()))
    execute(cmd)


    #from keras import backend as K
    #model = K.clear_session()


    import time
    time.sleep(30)


    results_txt.write("Time from start to now is (min): "+str((time.time()-t0)/60.0)+"\n")
    results_txt.close()


if __name__ == '__main__':
    parser = OptionParser(usage="Usage: %prog <random_state> <current_step_num>")
#    parser.add_option("-o", "--output", action="store", type="str", dest="outputFile", default=-1, help="Output FileName + location. Default is same location as Input.")
    options, args =  parser.parse_args()

    if len(args) < 2:
        parser.print_help()
        sys.exit(2)

    else:
        CnnModel(args)
        print ("DONE: CNN Model")
        t1 = time.time()
        total_time = t1-t0
        print ("total time: ",total_time/60.0,"   min")
