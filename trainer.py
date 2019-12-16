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
    dataset = dataset_loader(os.getcwd()+'/Data/dataset_alpha_121019.pkl')

    features = np.array([data['x_vector'] for data in dataset])
    labels = np.array([data['y'] for data in dataset])
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=13)
    print('Training Features Shape:', x_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', x_test.shape)
    print('Testing Labels Shape:', y_test.shape)
    
    
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
#    
#    
#    
#    '''
#    train set analysis
#    '''
#    #Mean Absolute Error
#    preds = rf.predict(x_train)
#    errors = abs(preds - y_train)
#    print('Mean Absolute Error:', round(np.mean(errors), 2))
#    
#    #Mean Absolute Percentage Error & Accuracy
#    mape = 100 * (errors / y_train)
#    accuracy = 100 - np.mean(mape)
#    print('Accuracy:', round(accuracy, 2), '%.')
#    
#    #Root Mean Squared Error
#    rmse = np.sqrt(mean_squared_error(y_train, preds))
#    print('Root Mean Squared Error :', round(rmse, 2))
#    
#    #Pearson Correlation Coefficient (PCC) score
#    pcc = pearsonr(y_train, preds)
#    print('Pearson Correlation Coefficient :', round(pcc[0],2))
#    print(preds, y_train)
#    
#    '''
#    test set analysis
#    '''
#    #Mean Absolute Error
#    preds = rf.predict(x_test)
#    errors = abs(preds - y_test)
#    print('Mean Absolute Error:', round(np.mean(errors), 2))
#    
#    #Mean Absolute Percentage Error & Accuracy
#    mape = 100 * (errors / y_test)
#    accuracy = 100 - np.mean(mape)
#    print('Accuracy:', round(accuracy, 2), '%.')
#    
#    #Root Mean Squared Error
#    rmse = np.sqrt(mean_squared_error(y_test, preds))
#    print('Root Mean Squared Error :', round(rmse, 2))
#    
#    #Pearson Correlation Coefficient (PCC) score
#    pcc = pearsonr(y_test, preds)
#    print('Pearson Correlation Coefficient :', round(pcc[0],2))
    

    '''
    k-fold cross validation
    '''
    folds = [3,4,5,7,10]
    for fold in folds:
        kfolds=[]
        n=fold
        idx = 0
        kf = KFold(n_splits=n)
        for train_index, test_index in kf.split(features):
            kfold = {}
            print("index training :",idx)
            print("TRAIN:", len(train_index), "TEST:", len(test_index))
            x_train, x_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            rf = RandomForestRegressor(n_estimators = 1000, random_state=13, verbose=0)
            rf.fit(x_train, y_train)

            idx+=1
            
            #Pearson Correlation Coefficient (PCC) score
            preds = rf.predict(x_train)
            pcc = pearsonr(y_train, preds)
            kfold["pcc_train"] = pcc[0]
            print('PCC train :', round(pcc[0],2))
            
            preds = rf.predict(x_test)
            pcc = pearsonr(y_test, preds)
            kfold["pcc_test"] = pcc[0]
            print('PCC test :', round(pcc[0],2))
            print('===================')
            
            kfold["train_idx"] = train_index
            kfold["test_idx"] = test_index
            kfold["k"] = n
            kfold["idx"] = idx
            kfold["model"] = rf
            kfolds.append(kfold)
        kfolds = sorted(kfolds, key=lambda k: k['pcc_test'], reverse=True) 
        print(kfolds[0]['k'], kfolds[0]['pcc_test'])
        #save best model
        with open(os.getcwd()+"/Model/rf_pp_a_"+str(n)+"fold_best.pkl", "wb") as f:
            pickle.dump(kfolds[0], f)
        
#    '''
#    model loader
#    '''
#    with open(os.getcwd()+"/Model/rf_pp_alpha.pkl", "rb") as f:
#        rf = pickle.load(f)