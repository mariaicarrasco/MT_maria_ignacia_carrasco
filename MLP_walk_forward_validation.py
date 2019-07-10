# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:44:23 2019

@author: Acer
"""

#MODELO FINAL: MLP
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam#SGD
from keras import losses, metrics
from hyperopt import hp, tpe, fmin, Trials
from keras.models import load_model
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

def evaluate_week(col_Y, n_week, n_eval):#(n_week, n_eval, col_Y):
    model2 = load_model('C:/Users/Acer/Documents/GitHub/MT/MLP/'+col_Y+'_'+str(n_week)+'_'+str(n_eval)+'.h5')
    y_pred = model2.predict(X_test) 
    y_train_pred = model2.predict(X_train)
    #Training Metrics
    r2Score_train[n_week] = r2_score(y_train, y_train_pred) 
    mse_train[n_week] = mean_squared_error(pd.DataFrame(y_train).values,y_train_pred)
    explained_var_train[n_week] = explained_variance_score(y_train,y_train_pred)
    poisson_loss_train[n_week] = model2.evaluate(X_train, y_train)[0]
    #Validation metrics 
    r2Score[n_week] = r2_score(y_test, y_pred) 
    mse[n_week] = mean_squared_error(pd.DataFrame(y_test).values,y_pred)
    explained_var[n_week] = explained_variance_score(y_test, y_pred)
    poisson_loss[n_week] = model2.evaluate(X_test, y_test)[0]
    return(r2Score_train, r2Score, mse_train, mse, explained_var_train, explained_var, poisson_loss_train, poisson_loss)
 
def evaluate_model (col_Y, r2Score_train, r2Score, mse_train, mse, explained_var_train, explained_var, poisson_loss_train, poisson_loss): 
    #Training Metrics    
    model_metrics['r2Score_train (mean)'][col_Y] = np.nanmean(r2Score_train)
    model_metrics['mse_train (mean)'][col_Y] = np.nanmean(mse_train)
    model_metrics['explained_var_train (mean)'][col_Y] = np.nanmean(explained_var_train)
    model_metrics['poisson_loss_train (mean)'][col_Y] = np.nanmean(poisson_loss_train)
    
    model_metrics['r2Score_train (deviation)'][col_Y] = np.nanstd(r2Score_train)
    model_metrics['mse_train (deviation)'][col_Y] = np.nanstd(mse_train)
    model_metrics['explained_var_train (deviation)'][col_Y] = np.nanstd(explained_var_train)
    model_metrics['poisson_loss_train (deviation)'][col_Y] = np.nanstd(poisson_loss_train)
    
    #Validation metrics  
    model_metrics['r2Score (mean)'][col_Y] = np.nanmean(r2Score)
    model_metrics['mse (mean)'][col_Y] = np.nanmean(mse)
    model_metrics['explained_var (mean)'][col_Y] = np.nanmean(explained_var)
    model_metrics['poisson_loss (mean)'][col_Y] = np.nanmean(poisson_loss)

    model_metrics['r2Score (deviation)'][col_Y] = np.nanstd(r2Score)
    model_metrics['mse (deviation)'][col_Y] = np.nanstd(mse)
    model_metrics['explained_var (deviation)'][col_Y] = np.nanstd(explained_var) 
    model_metrics['poisson_loss (deviation)'][col_Y] = np.nanstd(poisson_loss)
    
    print('TRAINING METRICS')
    print('R2 score:',np.nanmean(r2Score_train), np.nanstd(r2Score_train))
    print('Mean squared error:', np.nanmean(mse_train), np.nanstd(mse_train))
    print('Explained variance score:', np.nanmean(explained_var_train), np.nanstd(explained_var_train))
    print('Poisson loss:', np.nanmean(poisson_loss_train), np.nanstd(poisson_loss_train))
    
    print('VALIDATION METRICS')
    print('R2 score:',np.nanmean(r2Score), np.nanstd(r2Score))
    print('Mean squared error:', np.nanmean(mse), np.nanstd(mse))
    print('Explained variance score:', np.nanmean(explained_var), np.nanstd(explained_var)) 
    print('Poisson loss:', np.nanmean(poisson_loss), np.nanstd(poisson_loss))
    return(model_metrics)
    
#Define neural net structure
def objective(space):
    n_hidden_units = space['n_units']+1
    n_hidden_layers = space['n_layers']
    n_epoch = space['n_epochs']
    n_input = len(col_X)
    n_output = 1
    
    model = Sequential()
    model.add(Dense(n_input, input_dim=X_train.shape[1]))      
    for i in range (1, n_hidden_layers):
        model.add(Dense(n_hidden_units, activation='relu'))
        model.add(Dropout(0.2))#space['dropout_hidden']))
    model.add(Dense(n_output, activation='relu'))
    model.compile(optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),#SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
                  loss = losses.poisson, #objective function
                  metrics = [metrics.mean_squared_error])
    model.fit(X_train, y_train, epochs=n_epoch, validation_data=(X_test, y_test))    
    test_mse = model.evaluate(X_test, y_test) #return loss and metrics
    model.save('C:/Users/Acer/Documents/GitHub/MT/MLP/'+col_Y+'_'+str(j)+'_'+str(k[0])+'.h5')
    k[0]=+1
    y_pred = model.predict(X_test) 
    y_train_pred = model.predict(X_train)
    #Training Metrics
    r2Score_traink[k[0]] = r2_score(y_train, y_train_pred) 
    mse_traink[k[0]] = mean_squared_error(pd.DataFrame(y_train).values,y_train_pred)
    explained_var_traink[k[0]] = explained_variance_score(y_train,y_train_pred)
    poisson_loss_traink[k[0]] = model.evaluate(X_train, y_train)[0]
    #Validation metrics 
    r2Scorek[k[0]] = r2_score(y_test, y_pred) 
    msek[k[0]] = mean_squared_error(pd.DataFrame(y_test).values,y_pred)
    explained_vark[k[0]] = explained_variance_score(y_test, y_pred)
    poisson_lossk[k[0]] = model.evaluate(X_test, y_test)[0]
    return test_mse[0]

#load the dataset
df = pd.read_csv('C:/Users/Acer/Documents/GitHub/MT/MLP/BASE DE DATOS R_punto.csv', sep =';', header = 0, index_col = 0)
df = df.astype('float32')
col_X = df.columns[19:]
X = df[col_X]
col_y = df.columns[0:19].drop(['I_R','I_C','U_R','U_C'])
#Walk forward validation
n_train = len(X)-56
n_records = len(X)
aux = int((n_records-n_train)/7)
n_evals = 3

model_metrics = pd.DataFrame(columns=['poisson_loss (mean)', 'poisson_loss (deviation)', 'mse (mean)','mse (deviation)', 'r2Score (mean)', 'r2Score (deviation)', 'explained_var (mean)', 'explained_var (deviation)', 'poisson_loss_train (mean)', 'poisson_loss_train (deviation)', 'mse_train (mean)', 'mse_train (deviation)', 'r2Score_train (mean)', 'r2Score_train (deviation)', 'explained_var_train (mean)', 'explained_var_train (deviation)'], 
                                      index=col_y)
#%%
for col_Y in col_y:#df.columns[0:19]: 
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
        k=[0] #contador para las evaluaciones de tpe

        poisson_lossk = [np.nan for i in range(n_evals)]
        msek = [np.nan for i in range(n_evals)]
        r2Scorek = [np.nan for i in range(n_evals)]
        explained_vark = [np.nan for i in range(n_evals)]
        poisson_loss_traink = [np.nan for i in range(n_evals)]
        mse_traink = [np.nan for i in range(n_evals)]
        r2Score_traink = [np.nan for i in range(n_evals)]
        explained_var_traink = [np.nan for i in range(n_evals)]
        
        space = {'n_units': hp.randint('n_units', 17),
                 'n_layers':hp.randint('n_layers', 10)+1,
                 'n_epochs': hp.randint('n_epochs', 199)+1}
        
        trials = Trials()
        best_model = fmin(fn=objective,
                          space=space,
                          algo=tpe.suggest,
                          max_evals=n_evals,
                          trials=trials)
        
        best_models[j] = best_model, trials.losses(), trials.trials, k[0]
        min_loss = best_models[j][1][0] #variable auxiliar para saber cual de todas las evaluaciones del tpe fue la con menor perdida
        n_eval = 0
        for kk in range(2):#(n_evals):
            if (best_models[j][1][kk] < min_loss): 
                min_loss = best_models[j][1][kk]
                n_eval = kk
                print('n_semana=%i n_eval=%i' %(j,kk))
        week_metrics = evaluate_week(col_Y, j, n_eval)#(j, n_eval, col_Y)        
        j=j+1
    model_metrics = evaluate_model(col_Y, week_metrics[0], week_metrics[1], week_metrics[2], week_metrics[3], week_metrics[4], week_metrics[5], week_metrics[6], week_metrics[7])
    model_metrics.to_csv('C:/Users/Acer/Documents/GitHub/MT/MLP/model_metrics.csv', header = True, index = True)
