


import numpy as np
import os
import get_data as get
from tensorflow.keras import utils as np_utils
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt


import models as models

from eeg_reduction import eeg_reduction

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def step_decay(epoch):
    if(epoch < 20):
        lr = 0.01
    elif(epoch < 50):
        lr = 0.001
    else:
        lr = 0.001
    return lr
lrate = LearningRateScheduler(step_decay)


def save_results(history,num_classes,n_ds,n_ch,T,split_ctr):

   
    results = np.zeros((4,len(history.history['accuracy'])))
    results[0] = history.history['accuracy']
    results[1] = history.history['val_accuracy']
    results[2] = history.history['loss']
    results[3] = history.history['val_loss']
    results_str = os.path.join(results_dir,f'stats/global_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.csv')
                 
    np.savetxt(results_str, np.transpose(results))
    return results[0:2,-1]




experiment_name = 'your-global-experiment'

datapath = "/content/physionet.org/files/eegmmidb/1.0.0/"
results_dir=f'/content/eegnet-based-embedded-bci/results/'
#os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/stats', exist_ok=True)
os.makedirs(f'{results_dir}/model', exist_ok=True)
os.makedirs(f'{results_dir}/plots', exist_ok=True)


num_classes_list = [4] # list of number of classes to test {2,3,4}
n_epochs = 50 
n_ds = 1 
n_ch_list = [64] 
T_list = [3] 

# model settings 
kernLength = int(np.ceil(128/n_ds))
poolLength = int(np.ceil(8/n_ds))
num_splits = 5
acc = np.zeros((num_splits,2))


for num_classes in num_classes_list:
    for n_ch in n_ch_list:
        for T in T_list:

            # Load data
            X, y = get.get_data(datapath, n_classes=num_classes)

            ######## If you want to save the data after loading once from .edf (faster)
            np.savez(datapath+f'{num_classes}class',X_Train = X, y_Train = y)
            npzfile = np.load(datapath+f'{num_classes}class.npz')
            X, y = npzfile['X_Train'], npzfile['y_Train']

            # reduce EEG data (downsample, number of channels, time window)
            X = eeg_reduction(X,n_ds = n_ds, n_ch = n_ch, T = T)

            # Expand dimensions to match expected EEGNet input
            X = (np.expand_dims(X, axis=-1))
            # number of temporal sample per trial
            n_samples = np.shape(X)[2]
            # convert labels to one-hot encodings.
            y_cat = np_utils.to_categorical(y)

            # using 5 folds
            kf = KFold(n_splits = num_splits)

            split_ctr = 0
            for train, test in kf.split(X, y):
                
                # init model 
                model = models.DeepConvNet(nb_classes = num_classes, Chans=n_ch, Samples=n_samples,
                                dropoutRate=0.5,dropoutType='Dropout'
                              )
               
                #print(model.summary())

                # Set Learning Rate
                adam_alpha = Adam(lr=(0.001))
                model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])
                np.random.seed(42*(split_ctr+1))
                np.random.shuffle(train)
                # do training
                history = model.fit(X[train], y_cat[train], 
                        validation_data=(X[test], y_cat[test]),
                        batch_size = 16, epochs = n_epochs, callbacks=[lrate], verbose = 2)

                acc[split_ctr] = save_results(history,num_classes,n_ds,n_ch,T,split_ctr)
                
                print('Fold {:}\t{:.4f}\t{:.4f}'.format(split_ctr,acc[split_ctr,0], acc[split_ctr,1]))
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')

                #Save model
                model.save(os.path.join(results_dir,f'model/global_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.h5'))

                #Clear Models
                K.clear_session()
                split_ctr = split_ctr + 1
            plt.savefig('foo.png')
            print('AVG \t {:.4f}\t{:.4f}'.format(acc[:,0].mean(), acc[:,1].mean()))


           


