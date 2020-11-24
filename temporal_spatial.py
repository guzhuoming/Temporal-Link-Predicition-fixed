import pandas as pd
import numpy as np
import math
from math import sqrt
import csv
from keras.models import Sequential, Model
from keras.layers import Lambda, dot, Activation, concatenate, Input, Dense, Dropout, SimpleRNN, LSTM, GRU, Bidirectional, Layer
from keras import optimizers
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
# from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow import set_random_seed
from only_temporal import attention
set_random_seed(2)
name = open('./data/name.csv')
df_name = pd.read_csv(name)
name_node_pairs = df_name['name_node_pairs']

def node2vec_lstm(n_features=4,
         n_timesteps=12,
         n_train=10,
         n_window=5,
         n_units=100,
         n_epochs=50,
         with_att=False,
         lr=0.001
         ):
    """
    :param n_features: 4 or 10, using 4 features or 10 features
    :param n_train: training timesteps
    :param n_window: width of training window, for example, [0 1 2 3 4]->[5], n_window = 5
    :param n_units: LSTM units
    :param n_epochs: trainning epochs
    :return:
    """
    data = []

    for i in range(len(name_node_pairs)):
        f = open('./data/features_4_node2vec/{}_temp_link_ft.csv'.format(name_node_pairs[i]))
        df = pd.read_csv(f)
        data.append(df.values)

    data = np.array(data)
    n_samples, n_timesteps, n_features = data.shape
    scaled_data = data.reshape((n_samples, n_timesteps * n_features))
    # scaled_data = scaler.fit_transform(scaled_data)
    scaled_data = scaled_data.reshape((n_samples, n_timesteps, n_features))

    # define problem properties
    n_test = 12 - n_train

    inputs = Input(shape=(n_window, n_features))
    return_sequences = False
    if with_att==True:
        return_sequences = True
    att_in = Bidirectional(LSTM(n_units, input_shape=(n_window, n_features), return_sequences=return_sequences))(inputs)
    if with_att==True:
        att_out = attention()(att_in)
        outputs = Dense(1)(att_out)
    else:
        outputs = Dense(1)(att_in)

    model = Model(inputs, outputs)
    opt = optimizers.Adam(lr=lr)
    model.compile(loss='mse', optimizer=opt)

    for i in range(n_train-n_window):
        history = model.fit(scaled_data[:, i: i+n_window, :], scaled_data[:, i+n_window, 1], epochs=n_epochs)

    # make prediction
    inv_yhat = []
    for i in range(n_test):
        yhat = model.predict(scaled_data[:, n_train-n_window+i:n_train+i, :])
        inv_yhat.append(yhat)

    inv_yhat = np.array(inv_yhat)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))  # inv_yhat.shape:(3, 736, 1)
    inv_yhat = inv_yhat.reshape((inv_yhat.shape[0], inv_yhat.shape[1]))
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))  # inv_yhat.shape:(3, 736)
    inv_yhat = inv_yhat.T
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))  # inv_yhat.shape:(736, 3)

    inv_yhat = np.concatenate((scaled_data[:, n_train:, 0], inv_yhat), axis=1)
    inv_yhat = inv_yhat.reshape((n_samples, n_test, 2))
    inv_yhat = np.concatenate((inv_yhat, scaled_data[:, n_train:, 2:]), axis=2)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    inv_yhat = np.concatenate((scaled_data[:, :n_train, :], inv_yhat), axis=1)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    inv_yhat = inv_yhat.reshape(n_samples, n_timesteps * n_features)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    # inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat.reshape(n_samples, n_timesteps, n_features)
    inv_yhat[inv_yhat < 0] = 0  # transform negative values to zero
    prediction = inv_yhat[:, -3:, 1]
    prediction = prediction.reshape(prediction.shape[0], prediction.shape[1], 1)
    original = data[:, -3:, 1]
    original = original.reshape(original.shape[0], original.shape[1], 1)
    concat = np.concatenate((original, prediction), axis=2)
    print('concat.shape:{}'.format(concat.shape))
    np.set_printoptions(threshold=1e6)
    print('concat\n{}'.format(concat))
    concat = concat.reshape(concat.shape[0] * concat.shape[1], concat.shape[2])
    df = pd.DataFrame(concat)
    df.columns = ['original', 'prediction']
    df.to_csv('./data/node2vec_LSTM/prediction_node2vec_LSTM.csv', index=False)
    rmse = sqrt(mean_squared_error(inv_yhat[:, -3:, 1], data[:, -3:, 1]))
    print('rmse: {}'.format(rmse))
if __name__=='__main__':
    print('temporal_spatial')
    node2vec_lstm()