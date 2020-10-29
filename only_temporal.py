import pandas as pd
import numpy as np
from numpy import concatenate
import math
from math import sqrt
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

name = open('./data/name.csv')
df_name = pd.read_csv(name)
name_node_pairs = df_name['name_node_pairs']

def la_ha():
    for i in range(len(name_node_pairs)):
        f = open('./data/LA_HA/{}_prediction.csv'.format(name_node_pairs[i]), 'w', newline='')
        csvwriter = csv.writer(f)
        csvwriter.writerow(['t', 'tran_sum_real', 'tran_sum_la', 'tran_sum_ha', 'difference_la', 'difference_ha'])
        for j in range(3):
            csvwriter.writerow([j+9, 0., 0., 0., 0., 0.])
        f.close()

    mse_la = 0
    mse_ha = 0
    for i in range(len(name_node_pairs)):
        f1 = open('./data/features_10/{}_temp_link_ft.csv'.format(name_node_pairs[i]))
        df_node_pair = pd.read_csv(f1)
        f1.close()

        f2 = open('./data/LA_HA/{}_prediction.csv'.format(name_node_pairs[i]))
        df_prediction = pd.read_csv(f2)
        f2.close()

        last_value = 0.
        historical_sum = 0.
        tran_sum_acc = 0

        for j in range(9):
            tran_sum_acc = tran_sum_acc + df_node_pair['tran_sum'][j]

        last_value = df_node_pair['tran_sum'][8]

        for j in range(9, 12):
            tran_sum = df_node_pair['tran_sum'][j]
            df_prediction['tran_sum_real'][j - 9] = tran_sum
            df_prediction['tran_sum_ha'][j - 9] = historical_sum / j
            tran_sum_acc = tran_sum_acc + tran_sum
            historical_sum = historical_sum + df_prediction['tran_sum_ha'][j - 9]
            df_prediction['tran_sum_la'][j - 9] = last_value
            df_prediction['difference_ha'][j - 9] = df_prediction['tran_sum_ha'][j - 9] - \
                                                    df_prediction['tran_sum_real'][j - 9]
            df_prediction['difference_la'][j - 9] = df_prediction['tran_sum_la'][j - 9] - \
                                                    df_prediction['tran_sum_real'][j - 9]

            # calculate mse in the loop, accumulate it outside the loop
            mse_ha = mse_ha + math.pow(df_prediction['difference_ha'][j - 9], 2)
            mse_la = mse_la + math.pow(df_prediction['difference_la'][j - 9], 2)

        df_prediction.to_csv('./data/LA_HA/{}.csv'.format(name_node_pairs[i]),index=False)
    rmse_ha = math.sqrt(mse_ha / (len(name_node_pairs) * 3))
    rmse_la = math.sqrt(mse_la / (len(name_node_pairs) * 3))
    print('rmse_ha:{}, rmse_la:{}'.format(rmse_ha, rmse_la))

# def arima():
#
def lstm():
    data = []
    for i in range(len(name_node_pairs)):
        f = open('./data/features_4/{}_temp_link_ft.csv'.format(name_node_pairs[i]))
        df = pd.read_csv(f)
        data.append(df.values)
    data = np.array(data)
    print('data: {}, \ndata.shape(): {}'.format(data, data.shape))

    # define train, test
    scaler = MinMaxScaler(feature_range=(0, 1))
    n_samples, n_timesteps, n_features = data.shape
    scaled_data = data.reshape((n_samples, n_timesteps*n_features))
    scaled_data = scaler.fit_transform(scaled_data)
    scaled_data = scaled_data.reshape((n_samples, n_timesteps, n_features))

    # define problem properties
    n_units = 100
    n_features = 4
    n_window = 5
    n_train = 9
    n_test = 3

    # define LSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(n_units, input_shape=(n_window, n_features))))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    # fit network
    for i in range(n_train-n_window):
        history = model.fit(scaled_data[:, i: i+n_window, :], scaled_data[:, i+n_window, 1], epochs=100)
        # plot history
        # plt.plot(history.history['loss'])
        # plt.show()

    # make prediction
    inv_yhat = []
    for i in range(n_test):
        yhat = model.predict(scaled_data[:, n_train-n_window+i:n_train+i, :])
        inv_yhat.append(yhat)

    inv_yhat = np.array(inv_yhat)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape)) #inv_yhat.shape:(3, 736, 1)
    inv_yhat = inv_yhat.reshape((inv_yhat.shape[0], inv_yhat.shape[1]))
    print('inv_yhat.shape:{}'.format(inv_yhat.shape)) #inv_yhat.shape:(3, 736)
    inv_yhat = inv_yhat.T
    print('inv_yhat.shape:{}'.format(inv_yhat.shape)) #inv_yhat.shape:(736, 3)

    inv_yhat = concatenate((scaled_data[:, n_train:, 0], inv_yhat), axis=1)
    inv_yhat = inv_yhat.reshape((n_samples, n_test, 2))
    inv_yhat = concatenate((inv_yhat, scaled_data[:, n_train:, 2:]), axis=2)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    inv_yhat = concatenate((scaled_data[:, :n_train, :], inv_yhat), axis=1)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    inv_yhat = inv_yhat.reshape(n_samples, n_timesteps*n_features)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat.reshape(n_samples, n_timesteps, n_features)
    rmse = sqrt(mean_squared_error(inv_yhat[:, -3:, 1], data[:, -3:, 1]))
    print('rmse: {}'.format(rmse))


if __name__=='__main__':
    print('only_temporal')
    la_ha()
    # lstm()