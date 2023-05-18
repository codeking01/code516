import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def generate_model_data(data_path, alpha, days):
    df = pd.read_csv(data_path)
    train_day = int((len(df['利率风险']) - days + 1))
    for property in ['上证收益率', '上证对数收益率', '上证指数日波动率', '短期流动性利差', '利率风险', '利率期限结构',
                     "总溢出指数"]:
        df[property] = scaler.fit_transform(np.reshape(np.array(df[property]), (-1, 1)))
    X_data, Y_data = list(), list()

    # 生成时序数据
    for i in range(train_day):
        Y_data.append(df['利率风险'][i + days - 1])
        for j in range(days):
            for m in ['上证收益率', '上证对数收益率', '上证指数日波动率', '短期流动性利差', '利率风险', '利率期限结构']:
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


# def svm_model(x_train, y_train, x_test, y_test):
#     model = svm.SVC(kernel='linear', probability=True)
#     model.fit(x_train, y_train)
#     trainPredict = model.predict(x_train)
#     trainPredict = scaler.inverse_transform(trainPredict)
#     y_train = scaler.inverse_transform(np.reshape(y_train, (-1, 1)))
#     testPredict = model.predict(x_test)
#     testPredict = scaler.inverse_transform(testPredict)
#     y_test = scaler.inverse_transform(np.reshape(y_test, (-1, 1)))
#     x_train = (x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
#     x_test = (x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
#     y_train = (y_train.reshape(y_train.shape[0], y_train.shape[1] * y_train.shape[2]))
#     y_test = (y_test.reshape(y_test.shape[0], y_test.shape[1] * y_test.shape[2]))
#     return y_train, trainPredict, y_test, testPredict

# 使用回归模型
def svm_model(x_train, y_train, x_test, y_test):
    """
    x_train 是一个三维数组，形状为 (样本数量, 天数, 特征数量)。但是，svm.SVR 模型期望的输入是一个二维数组，其中每行代表一个样本，
    每列代表一个特征。因此，需要将 x_train 调整为二维数组。x_train.shape[0] 返回 x_train 数组的第一个维度的大小，即样本数量。
    -1 的意思是根据其他维度的大小自动计算该维度的大小，以保持数据总量不变。
    因此，x_train.reshape(x_train.shape[0], -1) 的作用是将 x_train 从三维数组转换为二维数组，其中每行代表一个样本。
    同样地，x_test 也需要进行相同的调整。
    """
    model = svm.SVR(kernel='linear')
    model.fit(x_train.reshape(x_train.shape[0], -1), y_train)  # Reshape x_train to 2D array
    trainPredict = model.predict(x_train.reshape(x_train.shape[0], -1))
    trainPredict = scaler.inverse_transform(np.reshape(trainPredict, (-1, 1)))
    y_train = scaler.inverse_transform(np.reshape(y_train, (-1, 1)))
    testPredict = model.predict(x_test.reshape(x_test.shape[0], -1))
    testPredict = scaler.inverse_transform(np.reshape(testPredict, (-1, 1)))
    y_test = scaler.inverse_transform(np.reshape(y_test, (-1, 1)))
    x_train = x_train.reshape(x_train.shape[0], -1)  # Reshape x_train back to 2D array
    x_test = x_test.reshape(x_test.shape[0], -1)  # Reshape x_test back to 2D array
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
    return y_train, trainPredict, y_test, testPredict


if __name__ == '__main__':
    data_path = '统计建模指标1.csv'
    days = 5
    alpha = 0.25
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train, Y_train, X_test, Y_test = generate_model_data('统计建模指标1.csv', alpha, days)
    train_Y, trainPredict, test_Y, testPredict = svm_model(X_train, Y_train, X_test, Y_test)
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(list(trainPredict[200:300]), 'grey', lw=1, label='预测值')
    plt.plot(list(train_Y[200:300]), '-.*', color='k', lw=1, label='真实值')
    plt.legend(loc='upper right', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.title('训练集', fontsize=25)
    plt.savefig(f'WX.jpg')
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
    plt.savefig(f'WC.jpg')
    plt.show()

    RMSE1, MAE1, MSE1, MAPE1, AMAPE1 = evaluate(train_Y, trainPredict)
    RMSE2, MAE2, MSE2, MAPE2, AMAPE2 = evaluate(test_Y, testPredict)
    print(RMSE1, MAE1, MSE1, MAPE1, AMAPE1)
    print(RMSE2, MAE2, MSE2, MAPE2, AMAPE2)
