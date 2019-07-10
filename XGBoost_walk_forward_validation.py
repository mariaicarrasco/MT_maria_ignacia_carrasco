# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:26:40 2019

@author: Acer
"""
#MODELO FINAL: XGBoost
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler#, StandardScaler
from hyperopt import hp, tpe, fmin, Trials
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error

def standarize_dataset (X_train, y_train, X_test, y_test, input_scaler=None, output_scaler=None):     
    if input_scaler is not None:
    	input_scaler.fit(X_train)
    	X_train = input_scaler.transform(X_train)
    	X_test = input_scaler.transform(X_test)
    if output_scaler is not None:
    	output_scaler.fit(y_train)
    	y_train = output_scaler.transform(y_train)
    	y_test = output_scaler.transform(y_test)
    if input_scaler is None:
        X_train = X_train.values
        X_test = X_test.values
    if output_scaler is None:
        y_train = y_train.values
        y_test = y_test.values
    return X_train, y_train, X_test, y_test

def evaluate_week(y_pred, y_train_pred, n_week):#(n_week, n_eval, col_Y):
    #Training Metrics
    r2Score_train[n_week] = r2_score(y_train, y_train_pred) 
    mse_train[n_week] = mean_squared_error(pd.DataFrame(y_train).values,y_train_pred)
    explained_var_train[n_week] = explained_variance_score(y_train,y_train_pred)
    #Validation metrics 
    r2Score[n_week] = r2_score(y_test, y_pred) 
    mse[n_week] = mean_squared_error(pd.DataFrame(y_test).values,y_pred)
    explained_var[n_week] = explained_variance_score(y_test, y_pred)
    return(r2Score_train, r2Score, mse_train, mse, explained_var_train, explained_var)
 
def evaluate_model (col_Y, r2Score_train, r2Score, mse_train, mse, explained_var_train, explained_var): 
    #Training Metrics    
    model_metrics['r2Score_train (mean)'][col_Y] = np.nanmean(r2Score_train)
    model_metrics['mse_train (mean)'][col_Y] = np.nanmean(mse_train)
    model_metrics['explained_var_train (mean)'][col_Y] = np.nanmean(explained_var_train)
    
    model_metrics['r2Score_train (deviation)'][col_Y] = np.nanstd(r2Score_train)
    model_metrics['mse_train (deviation)'][col_Y] = np.nanstd(mse_train)
    model_metrics['explained_var_train (deviation)'][col_Y] = np.nanstd(explained_var_train)
    
    #Validation metrics  
    model_metrics['r2Score (mean)'][col_Y] = np.nanmean(r2Score)
    model_metrics['mse (mean)'][col_Y] = np.nanmean(mse)
    model_metrics['explained_var (mean)'][col_Y] = np.nanmean(explained_var)

    model_metrics['r2Score (deviation)'][col_Y] = np.nanstd(r2Score)
    model_metrics['mse (deviation)'][col_Y] = np.nanstd(mse)
    model_metrics['explained_var (deviation)'][col_Y] = np.nanstd(explained_var) 
    
    print('TRAINING METRICS')
    print('R2 score:',np.nanmean(r2Score_train), np.nanstd(r2Score_train))
    print('Mean squared error:', np.nanmean(mse_train), np.nanstd(mse_train))
    print('Explained variance score:', np.nanmean(explained_var_train), np.nanstd(explained_var_train))

    print('VALIDATION METRICS')
    print('R2 score:',np.nanmean(r2Score), np.nanstd(r2Score))
    print('Mean squared error:', np.nanmean(mse), np.nanstd(mse))
    print('Explained variance score:', np.nanmean(explained_var), np.nanstd(explained_var)) 
    return(model_metrics)
    
def objective(params):
    print ("Training with params : ", params)
    dtrain=xgb.DMatrix(X_train, y_train)
    dtest=xgb.DMatrix(X_test, y_test)
    model=xgb.train(params,dtrain)
    y_pred=model.predict(dtest)
    y_train_pred=model.predict(dtrain)
    
    #Training Metrics
    r2Score_traink[k[0]] = r2_score(y_train, y_train_pred) 
    mse_traink[k[0]] = mean_squared_error(pd.DataFrame(y_train).values,y_train_pred)
    explained_var_traink[k[0]] = explained_variance_score(y_train,y_train_pred)
    #Validation metrics 
    r2Scorek[k[0]] = r2_score(y_test, y_pred) 
    msek[k[0]] = mean_squared_error(pd.DataFrame(y_test).values,y_pred)
    explained_vark[k[0]] = explained_variance_score(y_test, y_pred)
    return msek[k[0]]#msek[0]

#load the dataset
df = pd.read_csv('C:/Users/Acer/Documents/GitHub/MT/XGBoost/BASE DE DATOS R_punto.csv', sep =';', header = 0, index_col = 0)
df = df.astype('float32')
col_X = df.columns[19:]
X = df[col_X]
col_y = df.columns[0:19].drop(['I_R','I_C','U_R','U_C'])

#Walk forward validation
n_train = len(X)-56
n_records = len(X)
aux = int((n_records-n_train)/7)
n_evals = 3

model_metrics = pd.DataFrame(columns=['mse (mean)','mse (deviation)', 'r2Score (mean)', 'r2Score (deviation)', 'explained_var (mean)', 'explained_var (deviation)', 'poisson_loss_train (mean)', 'poisson_loss_train (deviation)', 'mse_train (mean)', 'mse_train (deviation)', 'r2Score_train (mean)', 'r2Score_train (deviation)', 'explained_var_train (mean)', 'explained_var_train (deviation)'], 
                                      index=col_y)
for col_Y in col_y:#df.columns[0:19]: 
    y = df[col_Y]
    j=0 #contador de semanas
    mse = [np.nan for i in range(aux)]
    r2Score = [np.nan for i in range(aux)]
    explained_var = [np.nan for i in range(aux)]
    mse_train = [np.nan for i in range(aux)]
    r2Score_train = [np.nan for i in range(aux)]
    explained_var_train = [np.nan for i in range(aux)]
    best_models = [np.nan for i in range(aux)]

    #XGBoost
    for i in range(n_train, n_records,7):
        X_train, X_test = X.iloc[0:i], X.iloc[i:i+7]
        y_train, y_test = y.iloc[0:i], y.iloc[i:i+7]
        #Standarize data 
        X_train, y_train, X_test, y_test = standarize_dataset(X_train, y_train, X_test, y_test, MinMaxScaler())   
        
        k=[0] #contador para las evaluaciones de tpe
        msek = [np.nan for i in range(n_evals)]
        r2Scorek = [np.nan for i in range(n_evals)]
        explained_vark = [np.nan for i in range(n_evals)]
        mse_traink = [np.nan for i in range(n_evals)]
        r2Score_traink = [np.nan for i in range(n_evals)]
        explained_var_traink = [np.nan for i in range(n_evals)]
        
        space = {'eta' : hp.quniform('eta', 0.001, 0.5, 0.025), #learning rate
                 'max_depth' : hp.randint('max_depth', 13)+1, #ok
                 'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
                 'subsample' : 1, #denotes the fraction of observations to be randomly samples for each tree. Since is a timeseries problem = 1
                 'gamma' : hp.quniform('gamma', 0, 1, 0.05), #the minimum loss reduction required to make a split.
                 'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05), #ok
                 'objective': 'count:poisson', #ok
                 #'eval_metric': 'poisson-nloglik',
                 'silent' : 1}
        
        trials = Trials() #Trials object where the history of search will be stored
        best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=n_evals)
        print(best)
        
        best_models[j] = best, trials.losses(), trials.trials, k[0]
        print('best_models', best_models)
     
        dtrain=xgb.DMatrix(X_train, y_train)
        dtest=xgb.DMatrix(X_test, y_test)
        model=xgb.train(best,dtrain)
        y_pred=model.predict(dtest)
        y_train_pred=model.predict(dtrain)
        
        week_metrics=evaluate_week(y_pred, y_train_pred, j)  
        j=j+1
    model_metrics = evaluate_model(col_Y, week_metrics[0], week_metrics[1], week_metrics[2], week_metrics[3], week_metrics[4], week_metrics[5])
model_metrics.to_csv('C:/Users/Acer/Documents/GitHub/MT/XGBoost/model_metrics.csv', header = True, index = True)

    