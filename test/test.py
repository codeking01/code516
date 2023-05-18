import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import math
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import matplotlib


def generate_model_data(data_path, alpha, days):
    df = pd.read_csv(data_path)
    train_day = int((len(df['I']) - days + 1))
    for property in ['SI', 'I', 'IC', 'SZF', 'SZB', 'SZS', "Total"]:
        df[property] = scaler.fit_transform(np.reshape(np.array(df[property]), (-1, 1)))
    X_data, Y_data = list(), list()

    # 生成时序数据
    for i in range(train_day):
        Y_data.append(df['I'][i + days - 1])
        for j in range(days):
            for m in ['SI', 'I', 'IC', 'SZF', 'SZB', 'SZS']:
                X_data.append(df[m][i + j])
    X_data = np.reshape(np.array(X_data), (-1, 6 * 5))  # 5表示特征数量*天数

    train_length = int(len(Y_data) * alpha)

    X_train = np.reshape(np.array(X_data[:train_length]), (len(X_data[:train_length]), days, 6))
    X_test = np.reshape(np.array(X_data[train_length:]), (len(X_data[train_length:]), days, 6))
    Y_train, Y_test = np.array(Y_data[:train_length]), np.array(Y_data[train_length:])
    return X_train, Y_train, X_test, Y_test


def calc_MAPE(real, predict):
    Score_MAPE = 0
    for i in range(len(predict[:, 0])):
        Score_MAPE += abs((predict[:, 0][i] - real[:, 0][i]) / real[:, 0][i])
    Score_MAPE = Score_MAPE * 100 / len(predict[:, 0])
    return Score_MAPE


def calc_AMAPE(real, predict):
    Score_AMAPE = 0
    Score_MAPE_DIV = sum(real[:, 0]) / len(real[:, 0])
    for i in range(len(predict[:, 0])):
        Score_AMAPE += abs((predict[:, 0][i] - real[:, 0][i]) / Score_MAPE_DIV)
    Score_AMAPE = Score_AMAPE * 100 / len(predict[:, 0])
    return Score_AMAPE


def evaluate(real, predict):
    RMSE = math.sqrt(mean_squared_error(real[:, 0], predict[:, 0]))
    MAE = mean_absolute_error(real[:, 0], predict[:, 0])
    MSE = mean_squared_error(real[:, 0], predict[:, 0])
    MAPE = calc_MAPE(real, predict)
    AMAPE = calc_AMAPE(real, predict)
    return RMSE, MAE, MSE, MAPE, AMAPE


def lstm_model(X_train, Y_train, X_test, Y_test):
    d = 0.01
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(units=30, return_sequences=False))
    model.add(Dropout(d))  # 建立的遗忘层
    # model.add(LSTM(units=10, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(d))#建立的遗忘层
    model.add(Dense(1, activation='tanh'))  # hard_sigmoid   tanh
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=300, batch_size=20, verbose=1)

    trainPredict = model.predict(X_train)
    trainPredict = scaler.inverse_transform(trainPredict)
    Y_train = scaler.inverse_transform(np.reshape(Y_train, (-1, 1)))

    testPredict = model.predict(X_test)
    testPredict = scaler.inverse_transform(testPredict)
    Y_test = scaler.inverse_transform(np.reshape(Y_test, (-1, 1)))

    return Y_train, trainPredict, Y_test, testPredict


if __name__ == '__main__':
    data_path = '../develop_data/lstm_data/无舆情测试数据.csv'
    days = 5
    alpha = 0.25
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train, Y_train, X_test, Y_test = generate_model_data('../develop_data/lstm_data/无舆情测试数据.csv', alpha, days)
    train_Y, trainPredict, test_Y, testPredict = lstm_model(X_train, Y_train, X_test, Y_test)
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(list(trainPredict[200:300]), 'grey', lw=1, label='预测值')
    plt.plot(list(train_Y[200:300]), '-.*', color='k', lw=1, label='真实值')
    plt.legend(loc='upper right', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.title('训练集', fontsize=25)
    plt.savefig('WX.jpg')
    plt.show()
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(list(testPredict[1000:1030]), color='grey', lw=1, label='预测值')
    plt.plot(list(test_Y[1000:1030]), '--', color='k', label='真实值')
    plt.legend(loc='upper right', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.title('测试集', fontsize=25)
    plt.savefig('WC.jpg')
    plt.show()

    RMSE1, MAE1, MSE1, MAPE1, AMAPE1 = evaluate(train_Y, trainPredict)
    RMSE2, MAE2, MSE2, MAPE2, AMAPE2 = evaluate(test_Y, testPredict)
    print(RMSE1, MAE1, MSE1, MAPE1, AMAPE1)
    print(RMSE2, MAE2, MSE2, MAPE2, AMAPE2)
