


from keras.models import Model
from keras.layers.core import Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D,AveragePooling2D

from keras.layers.convolutional import SeparableConv2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D, Dropout
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from keras.layers import DepthwiseConv2D
from tensorflow.keras.constraints import max_norm
from matplotlib import pyplot as plt
import tensorflow as tf




###################################################################
############## EEGnet model changed from loaded model #############
###################################################################

def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5,dropoutType='Dropout'):
 
    D=2

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(128, (1, 8), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1_1     =  Conv2D(64, (1, 8), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1_2     =  Conv2D(32, (1, 8), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)

    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
    block1       = dropoutType(dropoutRate)(block1)


    block1_1      = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1_1)
    block1_1       = BatchNormalization(axis = 1)(block1_1)
    block1_1       = Activation('elu')(block1_1)
    
    block1_1       = dropoutType(dropoutRate)(block1_1)


    block1_2       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1_2)
    block1_2       = BatchNormalization(axis = 1)(block1_2)
    block1_2       = Activation('elu')(block1_2)
    
    block1_2       = dropoutType(dropoutRate)(block1_2)

    concat1=tf.keras.layers.concatenate([block1,block1_1,block1_2])

    block      = AveragePooling2D(pool_size=(1, 4))(concat1)
    block1       = Conv2D(32, (1, 8), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block)
    block1_1     =  Conv2D(16, (1, 8), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block)
    block1_2     =  Conv2D(8, (1, 8), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block)

    concat2=tf.keras.layers.concatenate([block1,block1_1,block1_2])
    block1       = AveragePooling2D(pool_size=(1, 2))(concat2)
    blocka     =  Conv2D(8, (1, 8), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = AveragePooling2D(pool_size=(1, 2))(blocka)
    blocka     =  Conv2D(4, (1, 8), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = AveragePooling2D(pool_size=(1, 2))(blocka)
   


    

    
    
    flatten      = Flatten()(block1)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)
