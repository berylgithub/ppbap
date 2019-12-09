# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:08:41 2019

@author: Saint8312
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import pickle
import os

def dataset_loader(filepath):
    data = []
    try:
        with open(filepath, 'rb') as fr:
            try:
                while True:
                    data.append(pickle.load(fr))
            except EOFError:
                pass            
    except FileNotFoundError:
        print('File is not found')
    saved_ids = [d['id'] for d in data]
    return data




if __name__ == '__main__':
    '''
    load and split the dataset
    '''
    dataset = dataset_loader('dataset.pkl')

    features = np.array([data['x_vector'] for data in dataset])
    labels = np.array([data['y'] for data in dataset])
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=13)
    print('Training Features Shape:', x_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', x_test.shape)
    print('Testing Labels Shape:', y_test.shape)
#    
#    '''
#    data regression
#    '''
#    rf = RandomForestRegressor(n_estimators= 1000, random_state=11, verbose=1)
#    rf.fit(x_train, y_train)
#    
#    '''
#    model saver
#    '''
#    with open(os.getcwd()+"/Model/rf_pp_alpha.pkl", "wb") as f:
#        pickle.dump(rf, f)
        
    '''
    model loader
    '''
    with open(os.getcwd()+"/Model/rf_pp_alpha.pkl", "rb") as f:
        rf = pickle.load(f)
    
    '''
    train set analysis
    '''
    #Mean Absolute Error
    preds = rf.predict(x_train)
    errors = abs(preds - y_train)
    print('Mean Absolute Error:', round(np.mean(errors), 2))
    
    #Mean Absolute Percentage Error & Accuracy
    mape = 100 * (errors / y_train)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    
    #Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_train, preds))
    print('Root Mean Squared Error :', round(rmse, 2))
    
    #Pearson Correlation Coefficient (PCC) score
    pcc = pearsonr(y_train, preds)
    print('Pearson Correlation Coefficient :', round(pcc[0],2))
    print(preds, y_train)
    
    '''
    test set analysis
    '''
    #Mean Absolute Error
    preds = rf.predict(x_test)
    errors = abs(preds - y_test)
    print('Mean Absolute Error:', round(np.mean(errors), 2))
    
    #Mean Absolute Percentage Error & Accuracy
    mape = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    
    #Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print('Root Mean Squared Error :', round(rmse, 2))
    
    #Pearson Correlation Coefficient (PCC) score
    pcc = pearsonr(y_test, preds)
    print('Pearson Correlation Coefficient :', round(pcc[0],2))
    

#    for i in range(len(preds)):
#        print(preds[i], y_test[i])