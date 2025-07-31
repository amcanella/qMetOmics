
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from keras.models import Sequential
from keras import Input
from keras.layers import Dense, Flatten, Conv1D, Dropout, MaxPooling1D, LeakyReLU, BatchNormalization, GRU, Activation, Bidirectional, LSTM
from tensorflow.keras import Model

nfreq = 22473 #32_768 #52_234
nmet = 66
ndata = 10_000

def buildModel(config, nfreq):
  #WE ADD AN IDENTITY BLOCK BECAUSE THE INPUT AND OUTPUT HAS THE SAME SIZE THX TO THE SAME PADDINGS
  #Set the layers filling with the parameters
  #model = keras.Sequential()
  input = Input(shape = (nfreq,1))
  
  #X_shortcut = input_tensor
  #-----------CONV 1
  conv_1 = Conv1D(filters=config['units0'],
                    kernel_size = config['conv_kernel0'], activation = 'relu')(input) #, kernel_initializer='he_normal')) padding='same'

  #conv_1_b = Activation('relu')(conv_1)
  conv_1_c = Dropout(rate=config['dropout_conv_0'])(conv_1)
  #conv_1 = BatchNormalization()(conv_1)
  
  maxpool_1 = MaxPooling1D(pool_size = config['pool_size_0'])(conv_1_c)


  #------------CONV 2
  conv_2 = Conv1D(filters=config['units1'],
                    kernel_size = config['conv_kernel1'], activation = 'relu')(maxpool_1) #, padding='same', kernel_initializer='he_normal'
  #conv_2_b = Activation('relu')(conv_2)
  maxpool_2 = MaxPooling1D(pool_size = config['pool_size_1'])(conv_2)

  #----------CONV 3
  conv_3 = Conv1D(filters=config['units2'],
                    kernel_size = config['conv_kernel2'],
                    activation = 'relu')(maxpool_2) # kernel_initializer='he_normal')), padding='same', kernel_initializer='he_normal'
  
  #conv_3_b = Activation('relu')(conv_3)
  conv_3_c = Dropout(rate= config['dropout_conv_2'])(conv_3)
  #conv_3 = BatchNormalization()(conv_3)
  #conv_3 = Add()([X_shortcut, conv_3])
  
  maxpool_3 = MaxPooling1D(pool_size = config['pool_size_2'])(conv_3_c)

  flat = Flatten()(maxpool_3)
  #bi_lstm = Bidirectional(LSTM(config['LSTM'], return_sequences=False))(maxpool_3)
  #model.add(GRU(units=500))

  dense_1 = Dense(units= config['dense_units0'],
                    activation = 'relu') (flat) #kernel_initializer='he_normal'))

#add one densa mas
  output = Dense(nfreq, activation = LeakyReLU(negative_slope = config['activation_dense1'] ))(dense_1) #kernel_initializer='he_normal'))

  model = Model(input, output)
  
  model.compile(keras.optimizers.Adam(learning_rate = config['learning_rate']),
                loss = 'mean_squared_error',
                metrics = ['mae']) #E

  return model


'''
# %%
# @title Sequential model
def buildModel(config, nfreq, nmet):

  #Set the layers filling with the parameters
  model = Sequential()
  #-----------CONV 1 CHANGEEEEET TO INPUTTT DECIAAA 
  model.add(Conv1D(filters=config['units0'],kernel_size = config['conv_kernel0'], activation = 'relu', input_shape = (nfreq,1))) #padding='same', kernel_initializer='he_normal'))


  model.add(Dropout(rate=config['dropout_conv_0']))
  #model.add(BatchNormalization())

  model.add(MaxPooling1D(pool_size = config['pool_size_0']))


  #------------CONV 2
  model.add(Conv1D(filters=config['units1'],
                    kernel_size = config['conv_kernel1'],
                    activation = 'relu')) #padding='same', kernel_initializer='he_normal'))


  model.add(MaxPooling1D(pool_size = config['pool_size_1']))

  #----------CONV 3
  model.add(Conv1D(filters=config['units2'], kernel_size = config['conv_kernel2'],activation = 'relu')) #    padding='same', kernel_initializer='he_normal'))

  model.add(Dropout(rate= config['dropout_conv_2']))
  #model.add(BatchNormalization())

  model.add(MaxPooling1D(pool_size = config['pool_size_2']))

  model.add(Flatten())

  #model.add(GRU(units=500))

  model.add(Dense(units= config['dense_units0'],
                    activation = 'relu')) #kernel_initializer='he_normal'))

  #add one densa mas
  model.add(Dense(nmet)) #, activation = keras.layers.LeakyReLU(negative_slope = config['activation_dense1'] ))) #kernel_initializer='he_normal'))

  model.compile(keras.optimizers.Adam(learning_rate = config['learning_rate']),
                loss = 'mean_squared_error',
                metrics = ['mae']) #E

  return model'''