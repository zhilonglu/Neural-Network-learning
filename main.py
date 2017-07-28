# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:12:02 2017

@author: RunningZ
"""

import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_validation import train_test_split, KFold
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn import gaussian_process
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

import warnings
warnings.filterwarnings('ignore')
    
def get_model_list(task_name):
    # ensemble model list
    model_list, name_list = [], []

#    model_list.append(linear_model.LinearRegression())
#    name_list.append('LR')
#    
#    model_list.append(linear_model.SGDRegressor())
#    name_list.append('LR_SGD')
##    
#    model_list.append(linear_model.Lasso(alpha = 1.0))
#    name_list.append('Lasso')    
#    
#    model_list.append(linear_model.Ridge (alpha = 1.0))
#    name_list.append('Ridge')
#    
#    model_list.append(linear_model.LassoLars(alpha=.1))
#    name_list.append('LassoLars')
#    
#    model_list.append(linear_model.BayesianRidge())
#    name_list.append('BayesianRidge')    
#    
#    model_list.append(KernelRidge(alpha=1.0))
#    name_list.append('KernelRidge')
#    
#    model_list.append(gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1))
#    name_list.append('GaussianProcess')
#         
#         
#    model_list.append(KNeighborsRegressor(weights = 'uniform',n_neighbors=3))
#    name_list.append('KNN_unif')
#    
#    model_list.append(KNeighborsRegressor(weights = 'distance',n_neighbors=3))
#    name_list.append('KNN_dist')
    
#    model_list.append(SVR(kernel = 'linear', C = 1, gamma = 'auto', coef0 = 0, degree = 2))
#    name_list.append('SVM_linear')
#    
#    model_list.append(SVR(kernel = 'poly', C = 1, gamma = 'auto', coef0 = 0, degree = 2))
#    name_list.append('SVM_poly')
#    
#    model_list.append(SVR(kernel = 'rbf', C = 1, gamma = 'auto', coef0 = 0, degree = 2))
#    name_list.append('SVM_rbf')
    
#    model_list.append(DecisionTreeRegressor())
#    name_list.append('DT')
    
#    model_list.append(RandomForestRegressor(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0))
#    name_list.append('RF')
    
    
    if task_name == 'C1':
        n = 25
    elif task_name == 'C2':
        n = 50
    elif task_name == 'C3':
        n = 50
    elif task_name == 'C4':
        n = 100
    elif task_name == 'C5':
        n = 200
    model_list.append(ExtraTreesRegressor(n_estimators=n, max_depth=None, max_features='auto', min_samples_split=2, random_state=0))
    name_list.append('ET')
    

#    model_list.append(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0))
#    name_list.append('GBT')
#    
#    model_list.append(AdaBoostRegressor(DecisionTreeRegressor(max_depth=1),
#                          n_estimators=100, random_state=np.random.RandomState(1)))
#    name_list.append('AdaBoostRegressor1')    
#    
#    model_list.append(AdaBoostRegressor(DecisionTreeRegressor(max_depth=2),
#                          n_estimators=100, random_state=np.random.RandomState(2)))
#    name_list.append('AdaBoostRegressor2')
  
    
    return model_list, name_list


def my_regressor(model, X_train, Y_train, X_test, Y_test, ss_Y): 
    model.fit(X_train, Y_train)
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    
    Y_train = ss_Y.inverse_transform(Y_train)
    Y_train_pred = ss_Y.inverse_transform(Y_train_pred)
    Y_test = ss_Y.inverse_transform(Y_test)
    Y_test_pred = ss_Y.inverse_transform(Y_test_pred)
    
    score_train = my_score(Y_train, Y_train_pred)
    score_test = my_score(Y_test, Y_test_pred)
    
    return np.c_[score_train, score_test]    
    
def my_score(Y_real, Y_pred):
    #RMSE = np.sqrt(np.mean(np.power(Y_real - Y_pred, 2)))
    #MAE = np.mean(np.abs(Y_real - Y_pred))
    MAPE = np.mean(np.abs(np.ones_like(Y_pred) - Y_pred / Y_real))
    #return np.array([RMSE, MAE, MAPE]).reshape(-1, 1)
    return np.array([MAPE]).reshape(-1, 1)
    
    
def feature_extraction(X, task_name):
    # time series feature
    X_self = X.copy()
    
    # statical feature
    X_p0, X_p25, X_p50, X_p75, X_100 = np.percentile(X, (0, 25, 50, 75, 100),axis = 1)
    X_mean = np.mean(X, axis = 1)    
    X_sum = np.sum(X, axis = 1)
    X_std = np.std(X, axis = 1)
    X_var = np.var(X, axis = 1)
    X_diff = np.diff(X, axis = 1)
    X_diff2 = np.diff(X_diff, axis = 1)
    X_statical = np.c_[X_p0, X_p25, X_p50, X_p75, X_100, X_mean, X_sum, X_std, X_var]    
    
    
    # discrete feature
    X_int01 = (X / 1).astype(np.int)
    X_int05 = (X / 5).astype(np.int)
    X_int10 = (X / 10).astype(np.int)
    X_int20 = (X / 20).astype(np.int)
    X_int30 = (X / 30).astype(np.int)
    X_int40 = (X / 40).astype(np.int)
    X_int50 = (X / 50).astype(np.int)
    X_discrete = np.c_[X_int05]    
    
    # frequency feature
    n = X.shape[1]
    X_fft = abs(np.fft.fft(X, axis = 1)) / (n / 2.0);
    X_fft[0] = X_fft[0] / 2.0;
    X_fft = X_fft[:, : n / 2.0 + 1];
    X_frequency = np.c_[X_fft]
    
    
    if task_name == 'C1':
        return np.c_[X_statical, X_statical]
    elif task_name == 'C2':
        return np.c_[X_self, X_frequency]
    elif task_name == 'C3':
        return np.c_[X_self, X_statical, X_discrete, X_frequency]
    elif task_name == 'C4':
        return np.c_[X_discrete]
    elif task_name == 'C5':
        return np.c_[X_frequency]

def feature_selection(data, n_Y):
    ###########################################################################
    ## feature selection
    data = data[:, 24:] # exclue all weather
    #data = np.c_[data[:, 6:12], data[:, 18:24], data[:, 24:]] # exclue weather
    
    feature_corr = np.sum(np.abs(np.corrcoef(data.T)[-n_Y:,:-n_Y]), 0)
    #print(feature_corr)
    cumsum = 1.0 * np.cumsum(feature_corr[np.argsort(feature_corr)[::-1]]) / feature_corr.sum()
    index = np.searchsorted(cumsum, 0.8, 'left')
    index = max(5, index)
    #print(feature_corr)        
    #print(np.argsort(feature_corr)[-index:])    
    #data = np.c_[data[:, np.argsort(feature_corr)[-index:]], data[:, -12:]]    
    #print(cumsum)
    #print('remain feagure = %d / %d'%(len(feature_corr[feature_corr >= 3.6]), len(feature_corr)))
    #data = np.c_[data[:, feature_corr >= 3.6], data[:, -n_Y:]]
    ## end of feature selection
    ###########################################################################
    return data
    
def init_data(task_name):
    # init data
    #X = np.random.random((100, 12))
    #Y = np.random.random((100, 1))
    
    '''
    from sklearn.datasets import load_boston
    boston = load_boston()
    X = boston.data
    Y = boston.target
    Y = Y.reshape(len(Y), -1)
    Y = np.c_[Y, 2 * Y]
    '''
    
#    if 'am' in task_name:
#        data = np.loadtxt('data/%s/tensor45.csv'%(task_name), dtype = np.float64, delimiter=',')
#    elif 'pm' in task_name:
#        data = np.loadtxt('data/%s/tensor84.csv'%(task_name), dtype = np.float64, delimiter=',')
      
    data = np.loadtxt('data/%s/tensor.csv'%(task_name), dtype = np.float64, delimiter=',')
    n_Y = 12    
    n_final = 7
    print('data')  
    #print(data.shape)  
    ####################
    # feature extraction    
    data = np.c_[data[:, :24], feature_extraction(data[:, 24:36], task_name), data[:, 36:]]
    print(data.shape)
      
    ###################
    # feature selection 
    data = feature_selection(data, n_Y)
    print(data.shape)
    

    X = data[:-n_final, :-n_Y]
    Y = data[:-n_final, -n_Y:]
    X_final = data[-n_final:, :-n_Y]
    return X, Y, X_final
 
 
def my_split(X, Y, n_valid):
    if n_valid == 0:
        return X, X, Y, Y
    X_train, Y_train = X[:-n_valid,:], Y[:-n_valid, :]
    X_test, Y_test = X[-n_valid:, :], Y[-n_valid:, :]
    return X_train, X_test, Y_train, Y_test
  
def single_task(task_name, cur_time, n_ensemble):
    # init data
    X, Y, X_final = init_data(task_name)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.35, random_state = 1)
    X_train, X_test, Y_train, Y_test = my_split(X, Y, 0)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    # standardize 
    ss_X = StandardScaler()
    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)
    X_final = ss_X.transform(X_final)    

    
    Y_final = []    
    model_list, name_list , score_list = [], [], []
    
    # multi-output
    # multi-output
    # multi-output    
    n_Y = Y_train.size / len(Y_train)
    for i in range(n_Y):
        #print("====== Y%d ======"%(i + 1))
        ss_Y = StandardScaler()
        Y_train_i = ss_Y.fit_transform(Y_train[:, i])
        Y_test_i = ss_Y.transform(Y_test[:, i])        
            
        
        # get model list
        model_list_i, name_list_i = get_model_list(task_name)
       
        # train
        Y_final_i = []
        score_list_i = []
        for model in model_list_i:    
            score = my_regressor(model, X_train, Y_train_i, X_test, Y_test_i, ss_Y)
            score_list_i.append(score)
            Y_final_i.append(ss_Y.inverse_transform(model.predict(X_final)))
        
        
        
        ############################################################################
        # only use top N models
        top_score_list_i = np.sum(np.array(score_list_i).reshape(-1, 2), axis = 1)
        top_score_list_i_index = np.argsort(top_score_list_i)[:n_ensemble]
        #print(top_score_list_i_index)
        top_score_list_i = top_score_list_i[top_score_list_i_index]
        #print(top_score_list_i)

        
        model_list_i2, name_list_i2, score_list_i2, Y_final_i2 = [],[],[],[]
        for index in top_score_list_i_index:
            model_list_i2.append(model_list_i[index])
            name_list_i2.append(name_list_i[index])
            score_list_i2.append(score_list_i[index])
            Y_final_i2.append(Y_final_i[index])
        #############################################################################
        
        # print Yi result
        if False: # False
            print('train, test')
            i = 0
            for name, score in zip(name_list_i2, score_list_i2):
                i = i + 1
                print('%d###:\t%s'%(i, name))        
                print(score)
            print('')        
        
        model_list.append(model_list_i2)
        name_list.append(name_list_i2)
        score_list.append(score_list_i2)
        Y_final.append(Y_final_i2)
        #print("train+valid error @ Y%d = %.4f"%(i + 1, top_score_list_i.mean()))
        
        
    # an single task complete    
    train_valid_error = np.sum(score_list, axis = 3).mean()
    print("train+valid error ###### Y1~Y6 ###### = %.4f"%(train_valid_error))
    
    Y_final = np.array(Y_final)
    if not os.path.exists('result/%s/%s'%(cur_time, task_name)):
        os.makedirs('result/%s/%s'%(cur_time, task_name))
    for i in range(Y_final.shape[0]):
        np.savetxt('result/%s/%s/result_Y%d.csv'%(cur_time, task_name, i + 1), Y_final[i, :, :], fmt = '%.4f', delimiter = ',')
    
    
    result_i = (np.mean(Y_final, axis = 1).T).ravel()
    np.savetxt('result/%s/%s/result_all.csv'%(cur_time, task_name), result_i, fmt = '%.4f', delimiter = ',')
    return result_i, train_valid_error

def my_cross_validation(task_name, cur_time, n_ensemble):
    # init data
    X, Y, X_final = init_data(task_name)
    
    train_valid_error_list = []
    kf = KFold(X.shape[0], 5)    
    for index in kf:
        # split data
        X_train, X_test, Y_train, Y_test = X[index[0],:], X[index[1],:],Y[index[0],:], Y[index[1],:]
        #print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

        # standardize 
        ss_X = StandardScaler()
        X_train = ss_X.fit_transform(X_train)
        X_test = ss_X.transform(X_test)
        X_final = ss_X.transform(X_final)    

    
        Y_final = []    
        model_list, name_list , score_list = [], [], []
    
        # multi-output
        # multi-output
        # multi-output    
        n_Y = Y_train.size / len(Y_train)
        for i in range(n_Y):
            #print("====== Y%d ======"%(i + 1))
            ss_Y = StandardScaler()
            Y_train_i = ss_Y.fit_transform(Y_train[:, i])
            Y_test_i = ss_Y.transform(Y_test[:, i])        
                
            
            # get model list
            model_list_i, name_list_i = get_model_list(task_name)
           
            # train
            Y_final_i = []
            score_list_i = []
            for model in model_list_i:    
                score = my_regressor(model, X_train, Y_train_i, X_test, Y_test_i, ss_Y)
                score_list_i.append(score)
                Y_final_i.append(ss_Y.inverse_transform(model.predict(X_final)))
            
            
            
            ############################################################################
            # only use top N models
            top_score_list_i = np.sum(np.array(score_list_i).reshape(-1, 2), axis = 1)
            top_score_list_i_index = np.argsort(top_score_list_i)[:n_ensemble]
            #print(top_score_list_i_index)
            top_score_list_i = top_score_list_i[top_score_list_i_index]
            #print(top_score_list_i)
    
            
            model_list_i2, name_list_i2, score_list_i2, Y_final_i2 = [],[],[],[]
            for index in top_score_list_i_index:
                model_list_i2.append(model_list_i[index])
                name_list_i2.append(name_list_i[index])
                score_list_i2.append(score_list_i[index])
                Y_final_i2.append(Y_final_i[index])
            #############################################################################
            
            # print Yi result
            if False: # False
                print('train, test')
                i = 0
                for name, score in zip(name_list_i2, score_list_i2):
                    i = i + 1
                    print('%d###:\t%s'%(i, name))        
                    print(score)
                print('')        
            
            model_list.append(model_list_i2)
            name_list.append(name_list_i2)
            score_list.append(score_list_i2)
            Y_final.append(Y_final_i2)
            #print("train+valid error @ Y%d = %.4f"%(i + 1, top_score_list_i.mean()))
            
        
        # an single task complete    
        train_valid_error = np.sum(score_list, axis = 3).mean()
        #print("train+valid error ###### Y1~Y6 ###### = %.4f"%(train_valid_error))
        train_valid_error_list.append(train_valid_error)
    
    # 10 flod cross validation train+valid error of one single task
    train_valid_mean = np.array(train_valid_error_list).mean()
    print('10 fold mean train+valid error = %.4f'%(train_valid_mean))
    return train_valid_mean

def main():
    np.random.seed(0)
    # ensembling top N model    
    for n_ensemble in range(19, 20):            
        cur_time = str(int((time.time() * 1000) % 1e6))
#        task_name_list = ["1_0am", "1_1am", "2_0am", "3_0am", "3_1am",
#                          "1_0pm", "1_1pm", "2_0pm", "3_0pm", "3_1pm"]
        task_name_list = ["C1", "C2", "C3", "C4", "C5"]
        
        # for debug 
        #cur_time = 'test'      
        #task_name_list = ["1_0pm"]
#        
#        print('cur_time = %s, n_ensemble = %d'%(cur_time, n_ensemble))   
#        result, train_valid_error = [], []
#        for task_name in task_name_list:
#            print(task_name)
#            result_i, train_valid_error_i = single_task(task_name, cur_time, n_ensemble)
#            result.append(result_i)
#            train_valid_error.append(train_valid_error_i)
#            
#        
#        train_valid_error_mean = np.array(train_valid_error).mean()
#        print('\nall_train+valid error = %.4f'%(train_valid_error_mean))
#        np.savetxt('result/%s/result_all_%.4f.csv'%(cur_time, train_valid_error_mean),\
#            np.array(result).reshape(-1, 1), fmt = '%.4f', delimiter = ',')
#        os.rename('result/%s/'%(cur_time), 'result/%.4f_%02d_t%s/'%(train_valid_error_mean, n_ensemble, cur_time))


        
        # cross validation
        # cross validation
        train_valid_cross_validation = []
        for task_name in task_name_list:
            print(task_name)
#            if not task_name == 'C4': 
#                continue
            train_valid_cross_validation.append(my_cross_validation(task_name, cur_time, n_ensemble)) # cross validation
        print("all cross validation train+valid error = %.4f"%(np.array(train_valid_cross_validation).mean()))

if __name__ == '__main__':
    main()
    
#==============================================================================
# top01: 0.1810
# top04: 0.1655
# top07: 0.1629
# top07: 0.1718 扩展的
# top11: 0.1619
# top24: 0.1600
#==============================================================================
