"""Model builder for regression to estimate metabolite concentrations (nmet)."""
from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense, Flatten, Conv1D, Dropout, MaxPooling1D, LeakyReLU


def buildModel(config, nfreq, nmet):
  #Defines the quantification model

    input = Input(shape = (nfreq,1))

    #-----------CONV 1
    conv_1 = Conv1D(filters=config['units0'],
                    kernel_size = config['conv_kernel0'],
                    activation = 'relu')(input)
    conv_1_c = Dropout(rate=config['dropout_conv_0'])(conv_1)
    maxpool_1 = MaxPooling1D(pool_size = config['pool_size_0'])(conv_1_c)

    #------------CONV 2
    conv_2 = Conv1D(filters=config['units1'],
                    kernel_size = config['conv_kernel1'],
                    activation = 'relu')(maxpool_1)
    maxpool_2 = MaxPooling1D(pool_size = config['pool_size_1'])(conv_2)

    #----------CONV 3
    conv_3 = Conv1D(filters=config['units2'],
                     kernel_size = config['conv_kernel2'],
                    activation = 'relu')(maxpool_2)
    conv_3_c = Dropout(rate= config['dropout_conv_2'])(conv_3)
    maxpool_3 = MaxPooling1D(pool_size = config['pool_size_2'])(conv_3_c)

    flat = Flatten()(maxpool_3)

    #----------DENSE
    dense_1 = Dense(units= config['dense_units0'],
                    activation = 'relu') (flat)

    output = Dense(nmet, activation = LeakyReLU(negative_slope = config['activation_dense1'] ))(dense_1)

    model = Model(input, output)

    return model
