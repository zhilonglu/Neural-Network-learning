# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:59:41 2017

@author: RunningZ
"""

import numpy as np
import os
from sklearn.preprocessing import StandardScaler


## weather data
path_weather_train = 'data_raw/weather/weather (table 7)_training_update.csv'
path_weather_test = 'data_raw/weather/weather (table 7)_test1.csv'
# train
data = np.loadtxt(path_weather_train, dtype='str', delimiter=',')
data = data[1:, 1:-1].astype(np.float32) % 360
data = np.c_[data[data[:,0]==6,1:],data[data[:,0]==9,1:], data[data[:,0]==15,1:],data[data[:,0]==18,1:]]
data = data[-29:, :]
# test
data2 = np.loadtxt(path_weather_test, dtype='str', delimiter=',')
data2 = data2[1:, 1:-1].astype(np.float32) % 360
data2 = np.c_[data2[data2[:,0]==6,1:],data2[data2[:,0]==9,1:], data2[data2[:,0]==15,1:],data2[data2[:,0]==18,1:]]
data2 = data2[-29:, :]
# combine
data = np.r_[data, data2]
ss = StandardScaler()
data = ss.fit_transform(data)
#data[data>1.5] = 1.5
#data[data<-1.5] = -1.5
#np.savetxt('data_raw/weather/weather.csv', data, fmt = '%.4f', delimiter=',')


## flow data
path_flow_train = 'data_raw/forplot.csv'
path_flow_test = 'data_raw/forplottest1.csv'
flow = np.loadtxt(path_flow_train, dtype='str', delimiter=',')[:,2:]
flow2 = np.loadtxt(path_flow_test, dtype='str', delimiter=',')[:,2:]
# 19~24 -> 25~30
# 46~51 -> 52~57
flow = np.c_[flow[:,18:18+12],flow[:,45:45+12]].astype(np.float32)
flow2 = np.c_[flow2[:,18:18+12],flow2[:,45:45+12]].astype(np.float32)
flow = np.c_[flow[:,0:6],flow[:,12:18],flow[:,6:12],flow[:,18:24]]
flow2 = np.c_[flow2[:,0:6],flow2[:,12:18],flow2[:,6:12],flow2[:,18:24]]


###########################
## combine weather and flow
n_train = flow.shape[0] / 5
n_test = flow2.shape[0] / 5
for i in range(5):
    ff = np.r_[flow[i * n_train: (i + 1) * n_train, :], flow2[i * n_test: (i + 1) * n_test, :]]
    weather_flow = np.c_[data, ff]
    weather_flow = np.r_[weather_flow[:12,:], weather_flow[19:,:]] # delete Guoqingjie
    if not os.path.exists('data/C%d'%(i+1)):
        os.makedirs('data/C%d'%(i+1))
    np.savetxt('data/C%d/tensor_new.csv'%(i+1), weather_flow, fmt = '%.4f', delimiter=',')
    