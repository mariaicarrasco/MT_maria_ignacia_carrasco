# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:34:21 2019

@author: Acer
"""
#MODELO FINAL: GLM
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import re

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
    
#Define neural net structure
def objective(X_train, y_train, X_test, y_test, w_model):
    if w_model == 'Poisson':
        poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
        poisson_results = poisson_model.fit()
        print(poisson_results.summary())
        y_pred0 = poisson_model.predict(poisson_results.params, X_test)
        y_train_pred0 = poisson_model.predict(poisson_results.params, X_train)
        y_pred = np.floor(poisson_model.predict(poisson_results.params, X_test)) # IDEM = y_pred2 = poisson_results.predict(X_test)
        y_train_pred = np.floor(poisson_model.predict(poisson_results.params, X_train)) 
    else:
        negative_binomial_model = sm.GLM (y_train, X_train, family=sm.families.NegativeBinomial())
        negative_binomial_results = negative_binomial_model.fit()
        print(negative_binomial_results.summary())
        y_pred0 = negative_binomial_model.predict(negative_binomial_results.params, X_test)
        y_train_pred0 = negative_binomial_model.predict(negative_binomial_results.params, X_train)
        y_pred = np.floor(negative_binomial_model.predict(negative_binomial_results.params, X_test)) # IDEM = y_pred2 = poisson_results.predict(X_test)
        y_train_pred = np.floor(negative_binomial_model.predict(negative_binomial_results.params, X_train)) 
    return y_pred, y_train_pred

#load the dataset
df = pd.read_csv('C:/Users/Acer/Documents/GitHub/MT/GLM/BASE DE DATOS R_punto.csv', sep =';', header = 0, index_col = 0)
which_model = pd.read_excel('C:/Users/Acer/Documents/GitHub/MT/GLM/modelo_a_usar.xlsx', header=0, index_col=0) 
df = df.astype('float32')
col_X = df.columns[19:]
col_y = df.columns[0:19].drop(['I_R','I_C','U_R','U_C'])
r = re.compile(".*PM25")
newlist = list(filter(r.match, col_X)) #Elimino las columnas con MP25
col_X = [e for e in col_X if e not in newlist]

X = df[col_X]
#Walk forward validation
n_train = len(X)-56
n_records = len(X)
aux = int((n_records-n_train)/7)
n_evals = 3

model_metrics = pd.DataFrame(columns=['mse (mean)','mse (deviation)', 'r2Score (mean)', 'r2Score (deviation)', 'explained_var (mean)', 'explained_var (deviation)', 'poisson_loss_train (mean)', 'poisson_loss_train (deviation)', 'mse_train (mean)', 'mse_train (deviation)', 'r2Score_train (mean)', 'r2Score_train (deviation)', 'explained_var_train (mean)', 'explained_var_train (deviation)'], 
                                      index=col_y)

for col_Y in col_y: 
    y = df[col_Y]
    j=0 #contador de semanas
    poisson_loss = [np.nan for i in range(aux)]
    mse = [np.nan for i in range(aux)]
    r2Score = [np.nan for i in range(aux)]
    explained_var = [np.nan for i in range(aux)]
    poisson_loss_train = [np.nan for i in range(aux)]
    mse_train = [np.nan for i in range(aux)]
    r2Score_train = [np.nan for i in range(aux)]
    explained_var_train = [np.nan for i in range(aux)]
    best_models = [np.nan for i in range(aux)]
    
    for i in range(n_train, n_records,7):
        X_train, X_test = X.iloc[0:i], X.iloc[i:i+7]
        y_train, y_test = y.iloc[0:i], y.iloc[i:i+7]
    
        X_train, y_train, X_test, y_test = standarize_dataset(X_train, y_train, X_test, y_test, MinMaxScaler())
        y_pred, y_train_pred = objective(X_train, y_train, X_test, y_test, which_model['MP10'][col_Y])
        
        r2Score_train, r2Score, mse_train, mse, explained_var_train, explained_var = evaluate_week(y_pred, y_train_pred, j)     
        j=j+1
    model_metrics = evaluate_model(col_Y, r2Score_train, r2Score, mse_train, mse, explained_var_train, explained_var)
    model_metrics.to_csv('C:/Users/Acer/Documents/GitHub/MT/GLM/model_metrics.csv', header = True, index = True)
    print('MODEL METRICS')
    print(model_metrics)