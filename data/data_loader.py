import os
import numpy as np
import pandas as pd
import emd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils_NRU_RBN.tools import StandardScaler
from utils_NRU_RBN.timefeatures import time_features
import warnings
import math
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None,predecomp =True,stlseason=23):
        # size [seq_len, label_len, pred_len]
        # info
        """if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:"""
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.flag=flag
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.stlseason = stlseason
        self.predecomp=predecomp
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.features=='S':
            df_data =df_raw.iloc[:,-1:]
        elif self.features=='MS'or self.features=='MM':
            df_data =df_raw.iloc[:,1:]

        c = int(len(df_data) * 0.7)
        d = int(c + int(len(df_data)-c)/2)
        border1s = [0, c, d]
        border2s = [c, d,len(df_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.flag=="train":
            s_begin = index*1
        else:
            s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        if self.flag=="train":
            return math.floor((len(self.data_x) - self.seq_len- self.pred_len + 1)/1) 
        else:
            return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_ETT_day(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='D', cols=None,predecomp =True,stlseason=23):
        # size [seq_len, label_len, pred_len]
        # info
        """if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:"""
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.flag=flag
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.stlseason = stlseason
        self.predecomp=predecomp
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.features=='S':
            df_data =df_raw.iloc[:,-1:]
        elif self.features=='MS'or self.features=='MM':
            df_data =df_raw.iloc[:,1:]
        """data_start = (df_data.values!=0).argmax(axis=0)
        df_data =df_data[data_start[0]:]
        max=np.max(df_data.values.nonzero())
        df_data=df_data[:max]"""

        c = int(len(df_data) * 0.7)
        d = int(c + int(len(df_data)-c)/2)
        border1s = [0, c, d]
        border2s = [c, d,len(df_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        """if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]"""

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        """        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:"""
        
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.flag=="train":
            s_begin = index*1
        else:
            s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        #r_begin = s_end
        #r_end = r_begin + self.pred_len
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        if self.flag=="train":
            return math.floor((len(self.data_x) - self.seq_len- self.pred_len + 1)/1) 
        else:
            return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)