import numpy as np 
import pandas as pd 
import math 
from numpy import array
import matplotlib.pyplot as plt 


CORdata = pd.read_csv('coronavirus.csv')

from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense

def load_data(data, time_step=2, range_day=1, validate_rate=0.67):
    seq_length = time_step + range_day
    result = []
    for index in range(len(data) - seq_length + 1):
        result.append(data[index: index + seq_length])
    result = np.array(result)
    print('total data: ', result.shape)

    train_size = int(len(result) * validate_percent)
    train = result[:train_size, :]
    validate = result[train_size:, :]

    x_train = train[:, :time_step]
    y_train = train[:, time_step:]
    x_validate = validate[:, :time_step]
    y_validate = validate[:, time_step:]

    return [x_train, y_train, x_validate, y_validate]


def base_model(feature_len=3, range_day=3, input_shape=(8, 1)):
    model = Sequential()

    model.add(LSTM(units=100, return_sequences=False, input_shape=input_shape))

    model.add(RepeatVector(range_day))
    model.add(LSTM(200, return_sequences=True))

    model.add(TimeDistributed(Dense(units=feature_len, activation='linear')))

    return model

def seq2seq(feature_len=1, range_day=1, input_shape=(8, 1)):

    # Encoder
    encoder_inputs = Input(shape=input_shape) 
    encoder = LSTM(units=100, return_state=True,  name='encoder')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]

    # Decoder
    reshapor = Reshape((1, 100), name='reshapor')
    decoder = LSTM(units=100, return_sequences=True, return_state=True, name='decoder')

    # Densor
    densor_output = Dense(units=feature_len, activation='linear', name='output')

    inputs = reshapor(encoder_outputs)
    all_outputs = []



    for _ in range(range_day):
        outputs, h, c = decoder(inputs, initial_state=states)

        inputs = outputs
        states = [state_h, state_c]

        outputs = densor_output(outputs)
        all_outputs.append(outputs)

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    model = Model(inputs=encoder_inputs, outputs=decoder_outputs)

    return model

def normalize_data(data, scal, feature_len):
    minmaxscaler = scal.fit(data)
    normalize_data = minmaxscaler.transform(data)
    return normalize_data

from sklearn.preprocessing import MinMaxScaler

scal = MinMaxScaler(feature_range=(0, 1))
normdata = normalize_data(CORdata, scal,CORdata.shape[1])

x_train, y_train, x_validate, y_validate = load_data(normdata,time_step=3, range_day=4, validate_rate=0.5)
print('train data: ', x_train.shape, y_train.shape)
print('validate data: ', x_validate.shape, y_validate.shape)

from keras import backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Activation, TimeDistributed, Dropout, Lambda, RepeatVector, Input, Reshape
from keras.callbacks import ModelCheckpoint

input_shape = (3, data.shape[1])
model = seq2seq(data.shape[1], 4, input_shape)
model.compile(loss='mse', optimizer='adam',metrics=['acc'])
model.summary()

#Training 
trainedmodel = model.fit(x_train, y_train, batch_size=3, epochs=70)

#evaluating the model
train_error = model.evaluate(x=x_train, y=y_train, batch_size=3, verbose=0)
print('Train error: %.8f MSE' % (train_error[0])))

val_error = model.evaluate(x=x_validate, y=y_validate, batch_size=3, verbose=0)
print('Validation error: %.8f MSE' % (val_error[0]))

train_predict = model.predict(x_train)
val_predict = model.predict(x_validate)

#inversing the normalization

def inverse_normalize_data(data, scal):
    for i in range(len(data)):
        data[i] = scaler.inverse_transform(data[i])

    return data

train_predict = inverse_normalize_data(train_predict, scal)
y_train = inverse_normalize_data(y_train, scaler)
validate_predict = inverse_normalize_data(val_predict, scal)
y_validate = inverse_normalize_data(y_validate, scal)


#To be continued (some issues to solve)
