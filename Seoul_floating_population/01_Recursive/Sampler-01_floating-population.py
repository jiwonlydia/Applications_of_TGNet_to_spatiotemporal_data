import pandas as pd
import numpy as np
import os

train_dataset = np.load('../data_preprocessed/train_image_pop.npy')
test_dataset = np.load('../data_preprocessed/test_image_pop.npy')

print(train_dataset.shape)
print(test_dataset.shape)


# # Sampler
def sampler_stamp(data, stamp, lag, step, temp=False):

    num_row = len(data)
    data_x, data_y = [], []
    for idx in range(num_row-stamp*lag-step):
        y = np.array(data[stamp*lag+idx : stamp*lag+idx+step])
        if not temp:
            x = np.transpose(data[[stamp*i+idx for i in range(lag)],:], [1,2,0])
            data_x.append(x)
            y = np.transpose(y, [1,2,0])
        data_y.append(y)
    print("Sampler Return", np.shape(data_x), np.shape(data_y))
    
    if not temp:
        return np.array(data_x), np.array(data_y)
    else:
        return np.array(data_y)


# # Make Temporal Information
print("Train shape: ", np.shape(train_dataset), ", Test shape: ", np.shape(test_dataset))

# Setting Some Parameters 
num_train, num_test = np.shape(train_dataset)[0], np.shape(test_dataset)[0]
num_row = num_train + num_test
print('num_row: ', num_row)

### Initialize numpy array of temporal information (one-hot encoding)
datasets_min_30 = np.zeros([num_row, 48])
datasets_dow = np.zeros([num_row, 7])

# 더미화 하기
# 30 mins, and day-of-week index are calculated below
for i in range(num_row):
    idx_30 = int(int(i)%48)
    idx_dow = int(int(i/48)%7)
    datasets_min_30[i,idx_30] = 1
    datasets_dow[i, idx_dow] = 1
    
def train_test_split(data, idx):
    return data[:idx], data[idx:]


# Split Train & Test Period

train_index = num_train #144 # 120 days and 144 time index
min_30_train, min_30_test = train_test_split(datasets_min_30, train_index)
dow_train, dow_test = train_test_split(datasets_dow, train_index)

print(min_30_train.shape, min_30_test.shape)
print(dow_train.shape, dow_test.shape)


# # Save Final Data
def save_data_stamp(STAMP, LAG, STEP):
    # train 
    x_train, y_train = sampler_stamp(train_dataset, stamp=STAMP, lag=LAG, step=1, temp=False)    
    min_30_train_y = sampler_stamp(min_30_train, stamp=STAMP, lag=LAG, step=1, temp=True)
    dow_train_y = sampler_stamp(dow_train, stamp=STAMP, lag=LAG, step=1, temp=True)
    temporal_train = np.concatenate((dow_train_y, min_30_train_y), axis=-1)
    
    np.savez(f'./data/x_train_stamp{STAMP}_lag{LAG}.npz', x_train)
    np.savez(f'./data/y_train_stamp{STAMP}_lag{LAG}.npz', y_train)
    np.savez(f'./data/temporal_train_stamp{STAMP}_lag{LAG}.npz', temporal_train)
    
    # test
    x_test, y_test = sampler_stamp(test_dataset, stamp=STAMP, lag=LAG, step=STEP, temp=False)
    min_30_test_y = sampler_stamp(min_30_test, stamp=STAMP, lag=LAG, step=STEP, temp=True)
    dow_test_y = sampler_stamp(dow_test, stamp=STAMP, lag=LAG, step=STEP, temp=True)
    temporal_test = np.concatenate((dow_test_y, min_30_test_y), axis=-1)

    np.savez(f'./data/x_test_stamp{STAMP}_lag{LAG}_step{STEP}.npz', x_test)
    np.savez(f'./data/y_test_stamp{STAMP}_lag{LAG}_step{STEP}.npz', y_test)
    np.savez(f'./data/temporal_test_stamp{STAMP}_lag{LAG}_step{STEP}.npz', temporal_test)


time_unit = [0.5,1,2,3,4,6,8,12,24]
stamp_list = [int(2*i) for i in time_unit]
step_list = [int(48/i) for i in stamp_list]
lag_list = [2*(i+1) for i in range(12)]
if not os.path.exists('./data'):
    os.mkdir('./data')
for i in range(len(time_unit)): # time unit에 따라
    for j in range(len(lag_list)): # lag에 따라
        print(f'STAMP={stamp_list[i]}, LAG={lag_list[j]}, STEP={step_list[i]}')
        save_data_stamp(STAMP=stamp_list[i], LAG=lag_list[j], STEP=step_list[i])

