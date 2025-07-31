#Summer 2024
#Alonso Code

# %% Import packages

import tensorflow as tf # type: ignore
import os

from numba import cuda 
device = cuda.get_current_device()
device.reset()

print('CURRENT DEVICE', device)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('\nImporting packages...')
import numpy as np # type: ignore
import pandas as pd
#import keras_tuner
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Dropout, MaxPooling1D, BatchNormalization, GRU
from tensorflow.keras import Model
from keras.optimizers import Adam
import time 
import scipy

import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from keras_tuner.tuners import RandomSearch
from keras.models import load_model
from tensorflow.keras.optimizers import Adam

import os
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from simulator import process_data
from simulator import Simulator

from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

import functions 
from modelQuant import buildModel

import wandb
from wandb.integration.keras import WandbCallback, WandbModelCheckpoint, WandbMetricsLogger 
# Use wandb-core
wandb.require("core")
wandb.login()
#0d86c681267d7dfd1f045e1852d14b0cf629d6ad

# --------------------------------------------------------------------------------------------------------------------------------------

p = process_data('/imppc/labs/jadlab/amoran/tbOmicsRepo/Metabo_tables_14.xlsx')
d = p.create_dict()

s = Simulator(dictionary = d, met_data=p.met_data, clust_data = p.clust_data)
print(f'\nthis is a test: {p.met_data[0][6]}')

#%%Import data
nfreq = 22473 #32_768 #52_234
nmet = 66
ndata = 10_000
#run_number=50
#nnn= 13

#for value in range(58, 61):
# %%
start = time.perf_counter()
from_ = 19
to_ = 27
x , y = functions.customGenerator(from_, to_)
print(x.shape)
samples, values = x.shape
x = x.reshape(samples, values ,1) #not using in colab, is a tf convention, but otherwise obtaining error 
y = y.reshape(samples, 67,1)
stop = time.perf_counter()

timer = (stop - start) /60
print(f'time to load data was {timer} minutes')
# %%
print (f'\nThe shape of the datasets are {x.shape, y.shape}')
# %%Prepare data 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state=42) #90% training and 10% testing for many samples #75% 25% for low samples

#Get the 10% of the training set for validation
val_set = int(len(X_train)*0.1)

#Coger 200 para este caso
X_t = X_train[:-val_set]
x_val = X_train[-val_set:]
y_t = y_train[:-val_set]
y_val = y_train[-val_set:]

#SCALE DATA, OJO QUE NO SE SOBREESCRIBAN

y_t = functions.normalize_mets(y_t)
y_val = functions.normalize_mets(y_val)
y_test = functions.normalize_mets(y_test)

print(f'The shape of X_t is {X_t.shape} and the shape of y_t is {y_t.shape}')
# %%
best_hps = {'units0': 100,
'conv_kernel0': 36, #9,
#'units0B': 100,
#'conv_kernel0B': 36,
'dropout_conv_0': 0.1,
'BN': False,
'pool_size_0': 3,
'units1': 200,
'conv_kernel1': 20, #5,
#'units1B': 200,
#'conv_kernel1B': 20,
'pool_size_1': 3,
'units2': 500,
'conv_kernel2': 12, #3,
'dropout_conv_2': 0.1,
'pool_size_2':2,
'dense_units0': 200,
'activation_dense1': 0.05, #0.01 mejor?
'learning_rate': 0.0001,
'batch_size': 32, 
'batches': f'[{from_},{to_}]'}
#, 'LSTM': 64}

# Custom loss function with weighted MSE
def weighted_mse_loss(y_true, y_pred):
    # Define weights for each of the 66 outputs
    weights = tf.ones([66])

    # Update the value at position 33 (index 32) to 5.0
    indices = [[47]]  # 33rd position is at index 32
    updates = [10.0]
    weights = tf.tensor_scatter_nd_update(weights, indices, updates)
    
    # Calculate MSE for each output
    mse = tf.math.squared_difference(y_true, y_pred)
    
    # Multiply the MSE for each output by its corresponding weight
    weighted_mse = mse * weights
    
    # Return the mean of the weighted MSE
    return tf.reduce_mean(weighted_mse) #tensor



# %%Start run 

run_number = 270
nnn=1
run_name = f'SPURRR_customloss_10_{run_number}'
#run_name = f'RNN_LSTM_SPURR_90'
#run_name = f'EVAL_HIST_NEG005_{run_number}_{nnn}' 

ckpt_path = f'weightsCkpt/modelQuant/{run_name}_ckpt.weights.h5'
ckpt_dir = os.path.dirname(ckpt_path)
path_previous_w = f'weightsCkpt/modelQuant/SPURRR_customloss_10_180_ckpt.weights.h5' #f'weightsCkpt/modelQuant/no_leaky_180_ckpt.weights.h5' #f'weightsCkpt/modelQuant/EVAL_HIST_NEG005_7{run_number}_{nnn}_ckpt.weights.h5'


#run = wandb.init(project = 'trilasRNN', name = run_name, config = best_hps) #, id = 'r2fw818l', resume = 'allow') 
run = wandb.init(project ='tbOmics',  name = run_name, config = best_hps) #, id = '9gy2oa9p', resume = 'allow')

#Callbacks
wandb_call = WandbCallback(monitor = 'val_loss', mode = 'min', save_model = False )
logger = WandbMetricsLogger()
cp = keras.callbacks.ModelCheckpoint(filepath = ckpt_path ,monitor = 'val_loss', save_best_only= True, mode= 'min', save_weights_only=True, verbose=1)
lr_call = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25)
callbacks = [logger, cp]

#model function was here 

'''strategy = tf.distribute.MirroredStrategy()

print('\nNumber of devices!!: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope
with strategy.scope():'''
#%%
i_e = 1
t_e = 150

if i_e == 0:
  first = buildModel(best_hps, nfreq, nmet)
  #first.compile(keras.optimizers.Adam(learning_rate = best_hps['learning_rate']), loss = 'mean_squared_error', metrics = ['mae']) #E
  print('FIRST MODEL ')
else:
  #pass
  latest = tf.train.latest_checkpoint(ckpt_dir)
  print(f'\nLatest model is : {latest}')
  first = buildModel(best_hps, nfreq, nmet)
  
  first.load_weights(path_previous_w) #path_previous_w)
  print(f'LOAD WEIGHTS at: {path_previous_w}') #path_previous_w')
    

# Print the model summary
first.summary()

first.compile(Adam(learning_rate = best_hps['learning_rate']),
                loss = weighted_mse_loss,
                metrics = ['mae']) #E

history = first.fit(X_t, y_t, epochs = t_e, initial_epoch = i_e, validation_data = (x_val, y_val), callbacks=callbacks, verbose = 1, shuffle = True, batch_size = best_hps['batch_size'] ) #shuffle?? callback for log y poder plotear en wandb??

print(f'LOADING BEST CKPT...')
latest = tf.train.latest_checkpoint(ckpt_dir)
print(f'\nLatest model is : {latest}')
print(f'loadinggg: {ckpt_path}')
second = buildModel(best_hps, nfreq, nmet)
second.load_weights(ckpt_path)
second.compile(Adam(learning_rate = best_hps['learning_rate']),
                loss = weighted_mse_loss,
                metrics = ['mae'])

print('-----EVALUATION----')
t_loss, t_mae = second.evaluate(X_t, y_t)
test_loss, test_mae = second.evaluate(X_test, y_test)


pred_train = second.predict(X_t)
pred_test = second.predict(X_test)

print(f'the sizes are y_test: {y_test.shape} and pred_test {pred_test.shape}')
y_test = y_test.reshape(len(y_test), nmet)

print(f'\nreshaping y_test...{y_test.shape}')



r2 = r2_score(y_test, pred_test, multioutput='uniform_average')

print(f'\nThe R2 obtained was: {r2*100:.3f} % \n')

#To save r2 in wandb
wandb.config['R2'] = r2 #you can use wandb.log too to log the data 

l = []
for i in range(nmet):

  r = scipy.stats.pearsonr(y_test[:,i], pred_test[:,i])[0]
  print(f'met {i} with r: {r}')
  l.append(r)

print('Minimum correlation is: ', np.min(l))
print('Max corre is: ', np.max(l))
print(f'the mean of the correlations is {np.mean(l)}')

r_alanine_test = scipy.stats.pearsonr(y_test[:,2], pred_test[:,2])
r_histidine = scipy.stats.pearsonr(y_test[:,47], pred_test[:,47])
print(f'alanine correlation: {r_alanine_test[0]}')
print(f'glicine correlation: {scipy.stats.pearsonr(y_test[:,13], pred_test[:,13])[0]}')
wandb.config['r_ala'] = r_alanine_test[0]
wandb.config['r_hist'] = r_histidine[0]
wandb.config['r_mean'] = np.mean(l)


current_datetime = current_datetime= datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')

functions.plotGraphs(current_datetime, y_t, y_test, pred_train, pred_test, run_name)
functions.plotLossFunc(current_datetime, history, run_name)
functions.plotScatters(run_number, current_datetime, y_t, y_test, pred_train, pred_test,run_name)
functions.plotGraphTraining(current_datetime, y_t, pred_train,run_name)

print('------------------------------------\n')

#print(f'the keys ARE: {history.history.keys()}')  # Ensure 'loss' is one of the keys

#HABRA QUE HACER KERAS API TAL VEZ 
#print(history.history['loss'])
model_path=f'/imppc/labs/jadlab/amoran/tbOmicsRepo/models/{run_name}_66_bestGLYBIEN_{r2:.3f}.h5' 
second.save(model_path)
second.save(os.path.join(wandb.run.dir, f"model_{nmet}_best_{r2:.3f}.keras")) #no se guardo!!!

artifact = wandb.Artifact('model', type='model')
artifact.add_file(model_path)
wandb.log_artifact(artifact)

run.finish()