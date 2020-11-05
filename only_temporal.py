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
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from tensorflow import set_random_seed
set_random_seed(2)

name = open('./data/name.csv')
df_name = pd.read_csv(name)
name_node_pairs = df_name['name_node_pairs']



class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

# read embedding vector
def readEmbedding(rootdir):
    f = open(rootdir)
    line = f.readline()
    data_array = []
    while line:
        num = list(map(float, line.split(' ')))
        data_array.append(num)
        line = f.readline()
    f.close()
    # 736 128
    # 0 x x x x x x x x
    # 1 x x x x x x x x
    # 2 ......
    del data_array[0] # delete the first row
    data_array = list(map(lambda x:x[1:], data_array)) # delete the first column
    data_array = np.array(data_array)
    return data_array

def la_ha(n_train=10,

          ):
    for i in range(len(name_node_pairs)):
        f = open('./data/LA_HA/{}_prediction.csv'.format(name_node_pairs[i]), 'w', newline='')
        csvwriter = csv.writer(f)
        csvwriter.writerow(['t', 'tran_sum_real', 'tran_sum_la', 'tran_sum_ha', 'difference_la', 'difference_ha'])
        for j in range(12-n_train):
            csvwriter.writerow([j+n_train, 0., 0., 0., 0., 0.])
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

        for j in range(n_train):
            tran_sum_acc = tran_sum_acc + df_node_pair['tran_sum'][j]

        last_value = df_node_pair['tran_sum'][n_train-1]

        for j in range(n_train, 12):
            tran_sum = df_node_pair['tran_sum'][j]
            df_prediction['tran_sum_real'][j - n_train] = tran_sum
            df_prediction['tran_sum_ha'][j - n_train] = historical_sum / j
            tran_sum_acc = tran_sum_acc + tran_sum
            historical_sum = historical_sum + df_prediction['tran_sum_ha'][j - n_train]
            df_prediction['tran_sum_la'][j - n_train] = last_value
            df_prediction['difference_ha'][j - n_train] = df_prediction['tran_sum_ha'][j - n_train] - \
                                                    df_prediction['tran_sum_real'][j - n_train]
            df_prediction['difference_la'][j - n_train] = df_prediction['tran_sum_la'][j - n_train] - \
                                                    df_prediction['tran_sum_real'][j - n_train]

            # calculate mse in the loop, accumulate it outside the loop
            mse_ha = mse_ha + math.pow(df_prediction['difference_ha'][j - n_train], 2)
            mse_la = mse_la + math.pow(df_prediction['difference_la'][j - n_train], 2)

        df_prediction.to_csv('./data/LA_HA/{}.csv'.format(name_node_pairs[i]),index=False)
    rmse_ha = math.sqrt(mse_ha / (len(name_node_pairs) * (12-n_train)))
    rmse_la = math.sqrt(mse_la / (len(name_node_pairs) * (12-n_train)))
    print('rmse_ha:{}, rmse_la:{}'.format(rmse_ha, rmse_la))

# def arima():
#
def lstm(n_features=4,
         n_timesteps=12,
         n_train=10,
         n_window=5,
         n_units=100,
         n_epochs=50,
         with_att=False,
         methods='lstm',
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
        f = open('./data/features_{}/{}_temp_link_ft.csv'.format(n_features, name_node_pairs[i]))
        df = pd.read_csv(f)
        data.append(df.values)

    data = np.array(data)
    print('data: {}, \ndata.shape(): {}'.format(data, data.shape))

    # define train, test
    # scaler = MinMaxScaler(feature_range=(0, 1))
    n_samples, n_timesteps, n_features = data.shape
    scaled_data = data.reshape((n_samples, n_timesteps*n_features))
    # scaled_data = scaler.fit_transform(scaled_data)
    scaled_data = scaled_data.reshape((n_samples, n_timesteps, n_features))

    # define problem properties
    n_test = 12 - n_train

    # define LSTM
    # sequential
    # model = Sequential()
    # model.add(Bidirectional(LSTM(n_units, input_shape=(n_window, n_features))))
    # model.add(Dense(1))
    #
    # model.compile(loss='mse', optimizer='adam')

    # Model
    inputs = Input(shape=(n_window, n_features))
    return_sequences = False
    if with_att==True:
        return_sequences = True
    if methods=='lstm':
        att_in = Bidirectional(LSTM(n_units, input_shape=(n_window, n_features), return_sequences=return_sequences))(inputs)
    elif methods=='gru':
        att_in = Bidirectional(GRU(n_units, input_shape=(n_window, n_features), return_sequences=return_sequences))(inputs)
    elif methods=='rnn':
        att_in = Bidirectional(SimpleRNN(n_units, input_shape=(n_window, n_features), return_sequences=return_sequences))(inputs)
    if with_att==True:
        att_out = attention()(att_in)
        outputs = Dense(1)(att_out)
    else:
        outputs = Dense(1)(att_in)

    model = Model(inputs, outputs)
    opt = optimizers.Adam(lr=lr)
    model.compile(loss='mse', optimizer=opt)

    # fit network
    for i in range(n_train-n_window):
        history = model.fit(scaled_data[:, i: i+n_window, :], scaled_data[:, i+n_window, 1], epochs=n_epochs)
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

    inv_yhat = np.concatenate((scaled_data[:, n_train:, 0], inv_yhat), axis=1)
    inv_yhat = inv_yhat.reshape((n_samples, n_test, 2))
    inv_yhat = np.concatenate((inv_yhat, scaled_data[:, n_train:, 2:]), axis=2)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    inv_yhat = np.concatenate((scaled_data[:, :n_train, :], inv_yhat), axis=1)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    inv_yhat = inv_yhat.reshape(n_samples, n_timesteps*n_features)
    print('inv_yhat.shape:{}'.format(inv_yhat.shape))
    # inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat.reshape(n_samples, n_timesteps, n_features)
    inv_yhat[inv_yhat<0] = 0 # transform negative values to zero
    prediction = inv_yhat[:, -3:, 1]
    prediction = prediction.reshape(prediction.shape[0], prediction.shape[1], 1)
    original = data[:, -3:, 1]
    original = original.reshape(original.shape[0], original.shape[1], 1)
    concat = np.concatenate((original, prediction), axis=2)
    print('concat.shape:{}'.format(concat.shape))
    np.set_printoptions(threshold=1e6)
    print('concat\n{}'.format(concat))
    concat = concat.reshape(concat.shape[0]*concat.shape[1], concat.shape[2])
    df = pd.DataFrame(concat)
    df.columns = ['original', 'prediction']
    df.to_csv('./data/LSTM/prediction_LSTM_{}.csv'.format(n_features), index=False)
    rmse = sqrt(mean_squared_error(inv_yhat[:, -3:, 1], data[:, -3:, 1]))
    print('rmse: {}'.format(rmse))

def arima():
    mse = 0
    error = 0
    for i in range(len(name_node_pairs)):
        print(i)
        f = open('./data/features_4/{}_temp_link_ft.csv'.format(name_node_pairs[i]))
        df = pd.read_csv(f)

        data = df['tran_sum'].values
        train = data[0:10]
        history = [x for x in train]
        test = data[10:]
        pred = []
        try:
            for t in range(len(test)):
                model = ARIMA(history, order=(0,0,2))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                if yhat[0]<0:
                    yhat[0] = 0
                pred.append(yhat[0])
                history.append(test[t])
            print('pred_>=0:{}'.format(pred))
            mse = mse + mean_squared_error(test, pred)
            print(mse)
        except:
                error = error+1
                continue

    rmse = np.sqrt(mse/(len(name_node_pairs)-error))
    print('errornum:{}'.format(error))
    print('arima, rmse: {}'.format(rmse))

"""
random forest
"""
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[0]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
     # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]
# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, methods='randomforest'):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        if methods=='randomforest':
            yhat = random_forest_forecast(history, testX)
        elif methods=='xgboost':
            yhat = xgboost_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        # print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_squared_error(test[:, -1], predictions)
    return error, test[:, -1], predictions
def xgboost_forecast(train, testX):
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]

def randomforest():
    mse = 0
    for i in range(len(name_node_pairs)):
        print(i)
        f = open('./data/features_4/{}_temp_link_ft.csv'.format(name_node_pairs[i]))
        df = pd.read_csv(f)

        # load the dataset
        values = df['tran_sum'].values
        # transform the time series data into supervised learning
        data = series_to_supervised(values, n_in=6)
        # evaluate
        mse_, y, yhat = walk_forward_validation(data, 2, methods='randomforest')
        mse = mse+mse_
    rmse = np.sqrt(mse / len(name_node_pairs))
    print('randomforest, rmse: {}'.format(rmse))

def xgboost():
    mse = 0
    for i in range(len(name_node_pairs)):
        print(i)
        f = open('./data/features_4/{}_temp_link_ft.csv'.format(name_node_pairs[i]))
        df = pd.read_csv(f)

        # load the dataset
        values = df['tran_sum'].values
        # transform the time series data into supervised learning
        data = series_to_supervised(values, n_in=6)
        # evaluate
        mse_, y, yhat = walk_forward_validation(data, 2, methods='xgboost')
        mse = mse+mse_
    rmse = np.sqrt(mse / len(name_node_pairs))
    print('xgboost, rmse: {}'.format(rmse))

if __name__=='__main__':
    # print('only_temporal')
    # la_ha(n_train=10)
    # lstm(n_features=4, n_epochs=100, with_att=True, n_train=10, n_window=7, methods='lstm', lr=0.001)
    # arima()
    randomforest()