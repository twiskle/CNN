#!/usr/bin/python3
# Created by: Pchu (Oct 2018)

'''
Cnn_main

this script creates the Cnn model based on the steps as specified in the class CnnSteps

using Python 3.5.2

'''

##########################################################################

import os
from os import sys,path,system
import numpy as np
import CnnSteps

from sklearn.model_selection import StratifiedKFold

random_state = 7

print("random state is set to: {}".format(random_state))

#getting the largest dimension of the images for zero padding
#if there is a mismatch between the two set of images

x_Lsize = 256
y_Lsize = 256
z_Lsize = 193


save_path="/home/pchu/DATA/deep_learning/DTI_CTN_MSTN/Results"
file_suffix = "_DTI_1-param_epoch50_6fold_step"

##Dataset
'''
foldersPath1= ["/home/pchu/DATA/deep_learning/MM_CTN_MSTN_5d/MSTN/betted_trunc",
    "/home/pchu/DATA/deep_learning/MM_CTN_MSTN_5d/CTN/betted_trunc"]
'''

##mni_space

#foldersPath1= ["/home/pchu/DATA/deep_learning/T1_CTN_MSTN/MSTN/pons_masked_trunc", 
#    "/home/pchu/DATA/deep_learning/T1_CTN_MSTN/CTN/pons_masked_trunc"]


foldersPath1= ["/home/pchu/DATA/deep_learning/T1_CTN_MSTN/MSTN/brainstem_masked_trunc", 
    "/home/pchu/DATA/deep_learning/T1_CTN_MSTN/CTN/brainstem_masked_trunc"]

##subject_space
foldersPath2= ["/home/pchu/DATA/deep_learning/DTI_CTN_MSTN/MSTN/FA/betted", 
    "/home/pchu/DATA/deep_learning/DTI_CTN_MSTN/CTN/FA/betted"]

foldersPath3= ["/home/pchu/DATA/deep_learning/DTI_CTN_MSTN/MSTN/MD/betted", 
    "/home/pchu/DATA/deep_learning/DTI_CTN_MSTN/CTN/MD/betted"]

foldersPath4= ["/home/pchu/DATA/deep_learning/DTI_CTN_MSTN/MSTN/AD/betted", 
    "/home/pchu/DATA/deep_learning/DTI_CTN_MSTN/CTN/AD/betted"]


for step_num in range(1):#6):

    print("current step is: {}".format(step_num))

    model_suffix = file_suffix + str(step_num)

    import time
    t0 = time.time()

    #random_state, directories, x_Lsize, y_Lsize, z_Lsize
    Cnn = CnnSteps.CnnSteps(random_state,foldersPath1,x_Lsize,y_Lsize,z_Lsize)


    [X1,encoded_Y1] = Cnn.Pre_Model_Steps()
    
    ##4. create model
    batch_size = 4  ## or 128 / 256 depending on memory size
    epochs = 30
    num_classes = 2
    n_splits = 6


    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True,
        random_state=random_state)


    # Loop through the indices the split() method returns
    (index, (train_indices,val_indices)) = list(enumerate(
        kfold.split(X1, encoded_Y1)))[step_num]
    print("Training on fold {} / {} ...".format(step_num+1,kfold.n_splits))
    # Generate batches from indices
    x1train, x1val = X1[train_indices], X1[val_indices]


    x1train = x1train.astype('float32') / np.max(x1train)
    x1val = x1val.astype('float32') / np.max(x1val)


    y1train, y1val = encoded_Y1[train_indices], encoded_Y1[val_indices]


    # Clear model, and create it
    model = None

    #    from keras import backend as K
    #    tf_session = K.get_session()

    model = Cnn.FinalModel(num_classes,X1)#,X2,X3,X4)

    X2 = None       ## trying out 1 parameter first 
    X3 = None
    X4 = None


    ##Add Noise (Gaussian)
    noise_factor = 0.5
    x1train_noisy = x1train + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x1train.shape) 
    x1val_noisy = x1val + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x1val.shape) 

    x1train_noisy = np.clip(x1train_noisy, 0., 1.)
    x1val_noisy = np.clip(x1val_noisy, 0., 1.)

    # Debug message I guess
    print("Training new iteration on {} training samples, "
        "{} validation samples, this may be a while...".format(
        x1train.shape[0],x1val.shape[0]))
    print("Training indices: {}".format(train_indices))
    print("Validation indices: {}".format(val_indices))
    print("batch size: {}, epochs = {}".format(batch_size, epochs))


    '''
    train = model.fit(x1train, x1train, batch_size=batch_size,
        epochs=epochs,verbose=1, validation_data=(x1val, x1val))
    '''
    train = model.fit(x1train_noisy, x1train, batch_size=batch_size,
        epochs=epochs,verbose=1, validation_data=(x1val_noisy, x1val))



    ##7. Result and Figures

    if step_num < 1:
        results_txt = open(path.join(save_path,
            'results'+model_suffix+'.txt'),'w')
    else:
        results_txt = open(path.join(save_path,
            'results'+model_suffix+'.txt'),'a')

    import re
    results_txt.write("Stratified kfold, with number of folds: "
        "{} \n".format(kfold.n_splits))
    results_txt.write("Step {} \n".format(step_num))
    results_txt.write("batch size: {}, epochs = {} \n".format(
        batch_size, epochs))
    results_txt.write("Training indices: \n")
    results_txt.writelines(str(train_indices)+"\n")
    results_txt.write("Validation indices: \n")
    results_txt.writelines(str(val_indices)+"\n")
    results_txt.write(re.sub("[|]|'|,","",str(train.history.keys()))+"\n")
    results_txt.write(re.sub("[|]|'|,","",str(train.history.values()))+"\n")


    saveImageName="figure"+model_suffix+".png"
    Cnn.Result_Figure(train,save_path,saveImageName)


    ##8. Saving model
    model.save(path.join(save_path,"model"+model_suffix+".h5py"))


    ##9. Clear swap memory before next iteration
    cmd = ('sudo swapoff -a && sudo swapon -a').format(
        **dict(locals(), **globals()))
    execute(cmd)


    #from keras import backend as K
    #model = K.clear_session()

    import time
    time.sleep(30)

    results_txt.write("Time from start to now is (min): {} \n".format(
        (time.time()-t0)/60.0))
    results_txt.close()

    print ("DONE: CNN Model step {}".format(step_num))
    t1 = time.time()
    total_time = t1-t0
    print ("total time: ",total_time/60.0,"   min")


