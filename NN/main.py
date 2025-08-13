"""
Summer 2024
Author: Alonso
"""
#Import packages
import os
import numpy as np # type: ignore
import time 
import scipy
import datetime

import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from simulator import process_data
from simulator import Simulator
import functions 
from modelQuant import buildModel


NFREQ = 22473 #32_768
NMET = 66
ndata = 10_000

# --------------------------------------------------------------------------------------------------------------------------------------
#Import data from database
p = process_data('C:/path/to/database.xlsx')
d = p.create_dict()

s = Simulator(dictionary = d, met_data=p.met_data, clust_data = p.clust_data)
print(f'\nthis is a test: {p.met_data[0][6]}')


#Import simulated spectra in batches
start = time.perf_counter()
from_ = 19
to_ = 27
x , y = functions.custom_generator(from_, to_)

samples, values = x.shape
x = x.reshape(samples, values ,1) #not using in colab, is a tf convention, but otherwise obtaining error 
y = y.reshape(samples, 67,1)
stop = time.perf_counter()

timer = (stop - start) /60
print(f'time to load data was {timer} minutes')

# %%Prepare data 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state=42) #90% training and 10% testing for many samples #75% 25% for low samples

#Get the 10% of the training set for validation
val_set = int(len(X_train)*0.1)

X_t = X_train[:-val_set]
x_val = X_train[-val_set:]
y_t = y_train[:-val_set]
y_val = y_train[-val_set:]

#Scale data
y_t = functions.normalize_mets(y_t)
y_val = functions.normalize_mets(y_val)
y_test = functions.normalize_mets(y_test)

#Hyperparameters stored in dictionary
best_hps = {'units0': 100,
'conv_kernel0': 36,
'dropout_conv_0': 0.1,
'BN': False,
'pool_size_0': 3,
'units1': 200,
'conv_kernel1': 20,
'pool_size_1': 3,
'units2': 500,
'conv_kernel2': 12,
'dropout_conv_2': 0.1,
'pool_size_2':2,
'dense_units0': 200,
'activation_dense1': 0.05,
'learning_rate': 0.0001,
'batch_size': 32,
'batches': f'[{from_},{to_}]'}


# %%Start run 
model = buildModel(best_hps, NFREQ, NMET)

# Print the model summary
model.summary()

model.compile(optimizer = Adam(learning_rate = best_hps['learning_rate']),
                loss = 'mse',
                metrics = ['mae'])

i_e = 0
t_e = 150

#Train the model
history = model.fit(X_t, y_t, epochs = t_e, initial_epoch = i_e, validation_data = (x_val, y_val), verbose = 1, shuffle = True, batch_size = best_hps['batch_size'] )

#Evaluate the training
print('-----EVALUATION----')
t_loss, t_mae = model.evaluate(X_t, y_t)
test_loss, test_mae = model.evaluate(X_test, y_test)

pred_train = model.predict(X_t)
pred_test = model.predict(X_test)

y_test = y_test.reshape(len(y_test), NMET)
r2 = r2_score(y_test, pred_test, multioutput='uniform_average')
print(f'\nThe R2 obtained was: {r2*100:.3f} % \n')

l = []
for i in range(NMET):
  r = scipy.stats.pearsonr(y_test[:,i], pred_test[:,i])[0]
  print(f'met {i} with r: {r}')
  l.append(r)

print('Minimum correlation is: ', np.min(l))
print('Max corre is: ', np.max(l))
print(f'the mean of the correlations is {np.mean(l)}')

r_alanine_test = scipy.stats.pearsonr(y_test[:,2], pred_test[:,2])
print(f'alanine correlation: {r_alanine_test[0]}')
print(f'glicine correlation: {scipy.stats.pearsonr(y_test[:,13], pred_test[:,13])[0]}')

print('------------------------------------\n')

model_path='/path/to/model.h5'
model.save(model_path)
print('FINISHED!')