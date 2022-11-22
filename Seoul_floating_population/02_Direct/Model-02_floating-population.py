import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import keras
import keras.backend as K
import random
import math

from keras import layers
from keras.layers import Input, Dense, Conv2D, AveragePooling2D, Conv2DTranspose, Activation
from keras.layers import concatenate, BatchNormalization, Dropout, Add, RepeatVector, Reshape
from keras.models import Model
from keras import regularizers

#from tensorflow.keras.optimizers import SGD , Adam
from keras.optimizers import SGD , Adam
# from keras.utils.training_utils import multi_gpu_model


## Arguments

import argparse
parser = argparse.ArgumentParser(description='')

parser.add_argument('--stamp', type=int, default=1)
parser.add_argument('--lag', type=int, default=2)
parser.add_argument('--train', action='store_true')

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--decay', type=float, default=0.01)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--drop_p', type=float, default=0.1)
parser.add_argument('--reg', type=float, default=0.0)
parser.add_argument('--test', action='store_true')

parser.add_argument('--output_dir', type=str, default='./output/')
parser.add_argument('--save_dir', type=str, default='./model_saved/')
parser.add_argument('--model_name', type=str, default='no_named')

parser.add_argument('--scale', type=str, default='min_max')
parser.add_argument('--dataset_name', type=str, default='NYC')
parser.add_argument('--thr', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--num_gpu', type=int, default=0)

parser.add_argument('--temp', type=int, default=16)
parser.add_argument('--nf', type=int, default=32)
parser.add_argument('--enf', type=int, default=64)
parser.add_argument('--patience', type=int, default=150)
parser.add_argument('--es', type=str, default='min')

args, extras = parser.parse_known_args()

# # Load Data

# In[3]:


def load_np_data(filename):
    try:
        data = np.load(filename)['arr_0']
        print("[*] Success to load ", filename)
        return data
    except:
        raise IOError("Fail to load data", filename)

def get_min_max(data, scale='min_max'):
    if scale=='min_max':
        return np.min(data), np.max(data)
    else:
        return None, None
    
def min_max(data, min_value, max_value):
    result = data - min_value
    scale = max_value - min_value
    assert scale > 0
    result = result/scale
    return result

def scaler(data, scale_type='log', inv=False, min_value=None, max_value=None):
    if scale_type == 'log':
        if not inv:
            print("[*] ", np.shape(data), ":log scaled")
            return logscale(data)
        else:
            print("[*] ", np.shape(data), ": inverse log scaled")
            return inverse_logscale(data)
    elif scale_type == 'min_max':
        assert (min_value != None) and (max_value != None)
        if not inv:
            return min_max(data, min_value, max_value)
        else:
            return inverse_min_max(data, min_value, max_value)
    else:
        print("[!] invalid scale type: ", scale_type)
        raise
        
def inverse_min_max(data, min_value, max_value):
    scale = max_value - min_value
    result = scale * data + min_value
    return result


# In[4]:


def LoadData(STAMP, LAG, STEP, train=True, valid=False):
    
#     args.model_name = f'SeoulFloatingPop_lag{LAG}_step{STEP}'

    ### train set

    x_train = load_np_data(f'./data/x_train_stamp{STAMP}_lag{LAG}_step{STEP}_v2.npz')
    min_x, max_x = get_min_max(x_train, 'min_max')

    if train:    
        y_train = load_np_data(f'./data/y_train_stamp{STAMP}_lag{LAG}_step{STEP}_v2.npz')
        temporal_train = load_np_data(f'./data/temporal_train_stamp{STAMP}_lag{LAG}_step{STEP}_v2.npz')
        temporal_train = temporal_train.reshape(temporal_train.shape[0],temporal_train.shape[2])
        
        # Min Max Scaling
        x_train = scaler(x_train, 'min_max', inv=False, min_value=min_x, max_value=max_x)
        y_train = scaler(y_train, 'min_max', inv=False, min_value=min_x, max_value=max_x)
        
        ### validation set
        valid_ratio=0.2
        num_train = int(len(x_train)*(1.0-valid_ratio))
        x_train, x_valid = x_train[:num_train], x_train[num_train:]
        temporal_train, temporal_valid = temporal_train[:num_train], temporal_train[num_train:]
        y_train, y_valid = y_train[:num_train], y_train[num_train:]
        
        if valid:
            
            print(f'x_valid.shape = {x_valid.shape}')
            print(f'y_valid.shape = {y_valid.shape}')
            print(f'temporal_valid.shape = {temporal_valid.shape}')
        
            return x_valid, temporal_valid, y_valid
            
        else:
            
            print('--- training dataset ---')
            print(f'x_train.shape = {x_train.shape}')
            print(f'y_train.shape = {y_train.shape}')
            print(f'temporal_train.shape = {temporal_train.shape}')

            return x_train, temporal_train, y_train
    
    else: ### test set

        x_test = load_np_data(f'./data/x_test_stamp{STAMP}_lag{LAG}_step{STEP}_v2.npz')
        y_test = load_np_data(f'./data/y_test_stamp{STAMP}_lag{LAG}_step{STEP}_v2.npz')
        temporal_test = load_np_data(f'./data/temporal_test_stamp{STAMP}_lag{LAG}_step{STEP}_v2.npz')

        # Min Max Scaling
        x_test = scaler(x_test, 'min_max', inv=False, min_value=min_x, max_value=max_x)
        y_test = scaler(y_test, 'min_max', inv=False, min_value=min_x, max_value=max_x)

        print('--- test dataset ---')
        print(f'x_test.shape = {x_test.shape}')
        print(f'y_test.shape = {y_test.shape}')
        print(f'temporal_test.shape = {temporal_test.shape}')

        return x_test, temporal_test, y_test


# # TGNet

# In[5]:


def rmse(y_true, y_pred):
    rtn = np.sqrt(  np.average( np.square(y_pred-y_true) ) )
    return  rtn

def mape(y_true,y_pred):
    rtn = np.mean(np.abs((y_true - y_pred) / (1.0+y_true)))
    return rtn

def mape_trs(y_true,y_pred, trs=0):
    true_mask = y_true > trs
    tmp_abs = np.divide(np.abs(y_true-y_pred)[true_mask] , y_true[true_mask])

    rtn = (np.average(tmp_abs))
    return rtn

def rmse_trs(y_true,y_pred, trs=0):
    true_mask = y_true > trs
    tmp_abs = np.sqrt(np.average(np.square(y_pred-y_true)[true_mask]))
    return tmp_abs


# In[6]:


def gn_block(input, num_c=64, kernel_size=(3,3), strides=(1,1), padding='SAME', activation='relu', dropout=None, regularizer=0.01):
    net = AveragePooling2D(kernel_size, strides, padding)(input)
    net = Conv2D(num_c, kernel_size=(1,1), strides=strides, activation='linear', padding=padding, kernel_regularizer=regularizers.l1(regularizer))(net)

    net_sf = Conv2D(num_c, kernel_size=(1,1), strides=strides, activation='linear', padding=padding, kernel_regularizer=regularizers.l1(regularizer))(input)

    net = Add()([net, net_sf])
    net = concatenate([input, net])
    net = Conv2D(num_c, kernel_size=(1,1), strides=strides, activation=activation, padding=padding, kernel_regularizer=regularizers.l1(regularizer))(net)
    net = BatchNormalization()(net)

    if dropout == None:
        return net
    else:
        net = Dropout(dropout)(net)
        return net

def deconv_block(input, num_c=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', dropout=None, regularizer=0.01):
    net = Conv2DTranspose(num_c, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding, kernel_regularizer=regularizers.l1(regularizer))(input)
    net = BatchNormalization()(net)
    if dropout == None:
        return net
    else:
        net = Dropout(dropout)(net)
        return net


# In[7]:


def TGNet(input_shape):
    nf = args.nf
    h,w = input_shape[:2]
    start_input = Input(shape=input_shape)
    temporal_input = Input(shape=(55,))
    input_tensors = [start_input, temporal_input]


    ### Temporal guided embedding
    net_temp = Dense(args.temp, activation='relu')(temporal_input)
    # self.net_temp = Dense(args.temp, activation='relu')(net_temp)
    net_temp = RepeatVector(h*w)(net_temp)
    net_temp = Reshape((h,w,args.temp))(net_temp)

    ### U-net layers
    net1 = concatenate([start_input, net_temp], axis=-1)
    net1 = gn_block(net1, nf, dropout=args.drop_p,regularizer=args.reg)
    net11 = AveragePooling2D(pool_size=(2,2))(net1)
    net2 = gn_block(net11, nf*2,  dropout=args.drop_p, regularizer=args.reg)
    net3 = gn_block(net2, nf*2,  dropout=args.drop_p, regularizer=args.reg)
    net33 = concatenate([net2, net3])
    net4 = gn_block(net33, nf*2,  dropout=args.drop_p, regularizer=args.reg)
    net4 = concatenate([net2, net3, net4])

    net5 = deconv_block(net4, nf*4, (2,2), (2,2),   dropout=args.drop_p,regularizer=args.reg)
    net5 = concatenate([net5, net1])
    net6 = deconv_block(net5, nf*4, (3,3), (1,1), 'same',  dropout=args.drop_p, regularizer=args.reg)

    ## Position-wise Regression
    net7 = concatenate([net6, start_input, net_temp], axis=-1)
    net7 = gn_block(net7, nf*4, kernel_size=(1,1), dropout=args.drop_p, regularizer=args.reg)

    output = Conv2D(1, kernel_size=(1,1), padding='same', kernel_regularizer=regularizers.l2(args.reg))(net7)
    output = Activation('relu')(output)

    model = Model(inputs=input_tensors, outputs=output)

    return model


# In[8]:


def train(STAMP, LAG, STEP):

    x_train, temporal_train, y_train = LoadData(STAMP=STAMP, LAG=LAG, STEP=STEP, train=True, valid=False)
    x_valid, temporal_valid, y_valid = LoadData(STAMP=STAMP, LAG=LAG, STEP=STEP, train=True, valid=True)

    input_shape = [10, 20, LAG]
    model = TGNet(input_shape)

    model.compile(loss=['mean_absolute_error'], 
               optimizer=Adam(lr=0.001, decay=args.decay), 
               metrics=['mean_absolute_error'])

    
    print('===== START TRAINGING =====')
    
    best_eval_loss = 100000
    patience = 0

    for idx in range(args.epoch): # epoch
        if idx%100 == 0:
            print(f'Epoch = {idx}')
            
        with tf.device(f'/GPU:{args.num_gpu}'):
            model.fit([x_train, temporal_train], y_train, 
                  batch_size=args.batch, epochs=1, shuffle=True,verbose=0)

            eval_loss = model.evaluate([x_valid, temporal_valid], y_valid, 
                                   batch_size=args.batch, verbose=0)
        patience += 1
        if patience > args.patience:
            print(f'Epoch = {idx}, patience reached {args.patience}')
            break

        if best_eval_loss > eval_loss[-1]:
            if not os.path.exists('./model_saved'):
                os.mkdir('./model_saved')
            else:
                model.save(f'./model_saved/best_model_stamp{STAMP}_lag{LAG}_step{STEP}_v2.h5')

            best_eval_loss = eval_loss[-1]
            patience = 0
    print('===== END TRAINGING =====')


# In[9]:


def flatten_result(data):
    num_row, h, w = data.shape[:3]
    num_col = int(h*w)
    return np.reshape(data, [num_row, num_col])

def save_test_output(pred_inverse, y_inverse, output_path=None):
    num_row, h, w = pred_inverse.shape[:3]
    num_col = int(h*w)
    assert pred_inverse.shape[:3] == y_inverse.shape[:3]
    np_pred = flatten_result(pred_inverse) #np.reshape(pred_inverse, [num_row, num_col])
    np_y = flatten_result(y_inverse) #np.reshape(y_inverse, [num_row, num_col])

    col_name = ['col_'+str(i) for i in range(0, num_col)]
    index = np.arange(0, num_row)
    df_pred = pd.DataFrame(np_pred, columns=col_name, index=index)
    df_y = pd.DataFrame(np_y, columns=col_name, index=index)

    df_y.to_csv(output_path+'_gt.csv', index=False)
    df_pred.to_csv(output_path+'_pred.csv', index=False)


# # Forecasting

# In[10]:


def direct_forecast(STAMP, LAG, STEP):
    # Load saved model
    with tf.device(f'/GPU:{args.num_gpu}'):
        model = tf.keras.models.load_model(f'./model_saved/best_model_stamp{STAMP}_lag{LAG}_step{STEP}_v2.h5')
    
    min_x, max_x = get_min_max(load_np_data(f'./data/x_train_stamp{STAMP}_lag{LAG}_step{STEP}_v2.npz'), 'min_max')
    x_test, temporal_test, y_test = LoadData(STAMP=STAMP, LAG=LAG, STEP=STEP, train=False, valid=False)
    
    temporal_test_step1 = temporal_test[:,0,:]
    with tf.device(f'/GPU:{args.num_gpu}'):
        y_pred =  model.predict([x_test, temporal_test_step1])
    y_true = np.expand_dims(y_test[:,:,:,0], axis=-1)
    y_pred_inv = scaler(y_pred, 'min_max', inv=True, min_value=min_x, max_value=max_x)
    y_true_inv = scaler(y_true, 'min_max', inv=True, min_value=min_x, max_value=max_x)
    if not os.path.exists('./output'):
        os.mkdir('./output')
    save_test_output(y_pred_inv, y_true_inv, output_path=f'./output/stamp{STAMP}_lag{LAG}_step{STEP}_v2')
    RMSE = rmse(y_true_inv, y_pred_inv)
    print('#'*57)
    print(f'###  RMSE of STAMP {STAMP} LAG {LAG} STEP {STEP} = {RMSE} ###')
    print('#'*57)
    return RMSE


####### TRAIN & TEST #######
step = int(48/args.stamp)
if args.train == False:
    direct_forecast(STAMP=args.stamp, LAG=args.lag, STEP=step)
else: 
    train(STAMP=args.stamp, LAG=args.lag, STEP=step)
    direct_forecast(STAMP=args.stamp, LAG=args.lag, STEP=step)


# Parameter Tuning
# time_unit = [0.5,1,2,3,4,6,8,12,24]
# stamp_list = [int(2*i) for i in time_unit]
# step_list = [int(48/i) for i in stamp_list]
# lag_list = [2*(i+1) for i in range(12)]

# table = np.zeros(shape=(len(time_unit),len(lag_list)))
# for i in range(len(time_unit)): # time unit에 따라
#     for j in range(len(lag_list)): # lag에 따라
#         print(f'STAMP={stamp_list[i]}, LAG={lag_list[j]}, STEP={step_list[i]}')
#         try:
#             table[i,j] = single_step_forecast(STAMP=stamp_list[i], LAG=lag_list[j], STEP=step_list[i])
#         except:
#             train(STAMP=stamp_list[i], LAG=lag_list[j], STEP=step_list[i])
#             table[i,j] = single_step_forecast(STAMP=stamp_list[i], LAG=lag_list[j], STEP=step_list[i])
#         print(table)

# col_name = ['lag_'+str(i) for i in lag_list]
# index = [stamp_list, time_unit, step_list]
# rmse_table = pd.DataFrame(table, columns=col_name, index=index)
# display(rmse_table)
# rmse_table.to_csv(f'rmse_table_direct.csv')

