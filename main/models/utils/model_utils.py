# @author: code_king
# @date: 2023/7/26 20:06
import uuid

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def dimensionality_reduction(x_data=None, y_data=None, correlation_threshold=0.7):
    """
    :param y_data:
    :param x_data:
    :param correlation_threshold:相关性值阈值
    :return:
    """
    # 计算特征之间的相关性矩阵
    corr_matrix = np.abs(np.corrcoef(x_data, rowvar=False))
    # 设置相关性的阈值，根据需要进行调整
    correlation_threshold = correlation_threshold
    # 找到高度相关的特征索引
    highly_correlated_indices = np.where(corr_matrix > correlation_threshold)
    # 删除高度相关的特征
    features_to_drop = set()
    for i, j in zip(*highly_correlated_indices):
        if i != j:
            if i not in features_to_drop and j not in features_to_drop:
                # 判断两个特征中哪一个与目标变量相关性更低，保留相关性较低的特征
                corr_i_y = np.abs(np.corrcoef(x_data[:, i], y_data, rowvar=False)[0, 1])
                corr_j_y = np.abs(np.corrcoef(x_data[:, j], y_data, rowvar=False)[0, 1])
                if corr_i_y < corr_j_y:
                    features_to_drop.add(i)
                else:
                    features_to_drop.add(j)
    return features_to_drop


def batch_train(model=None, model_name=uuid.uuid4(), data_list=None, max_epochs=20, stop_accuracy=0.8, epochs=10,
                batch_size=32, features_to_drop=None):
    """
    :param model_name: 模型名称
    :param data_list: 训练集和测试集
    :param model:
    :param features_to_drop: 降维需要删除的列
    :param epochs:
    :param batch_size:
    :param max_epochs:  最大训练轮数
    :param stop_accuracy:  停止训练的准确率阈值
    :return:
    """
    accuracy = 0
    train_x, test_x, y_train, y_test = data_list
    for epoch in range(max_epochs):
        history_model = model.fit(train_x, y_train, epochs=epochs, batch_size=batch_size,
                                  validation_data=(test_x, y_test), verbose=0)
        # 模型评估
        pred_y_prob = model.predict(test_x)
        pred_value = (pred_y_prob > 0.5).astype("int32")
        accuracy = max(accuracy, metrics.accuracy_score(pred_value, y_test))
        print(f"Epoch: {epoch + 1} - Accuracy: {accuracy}")
        # print(classification_report(y_test, pred_value))
        # 当准确率超过阈值时保存模型
        if accuracy > stop_accuracy:
            models = {"features_to_drop": features_to_drop, "model": history_model}
            joblib.dump(models, f'../save_models/{model_name}.joblib')
            print("Model saved.")
            return history_model, pred_y_prob
    # print(Exception(f"Stop training, accuracy is too low, accuracy: {accuracy}"))
    raise Exception(f"Stop training, accuracy is too low, accuracy: {accuracy}")


def drawing(y_test=None, y_pred_prob=None, history=None):
    """
    :param history:
    :param y_test: 真实标签
    :param y_pred_prob: 预测概率
    :return:
    """
    # 计算ROC曲线
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    # 计算AUC值
    roc_auc = metrics.auc(fpr, tpr)
    # 绘制ROC曲线
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC')
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.plot(fpr, tpr, 'r', label=f'CNN AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend()
    plt.show()
    # 绘制训练曲线
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.show()


def get_data(filepath=None):
    data = pd.read_excel(f'{filepath}')
    # 读取和预处理数据
    data['ST是否'] = data['ST是否'].fillna('1')
    data['截止日期_Enddt'] = str(data['截止日期_Enddt'])
    data = data.dropna()
    for feature in data.columns:
        if data[feature].dtype == 'object':
            data[feature] = pd.Categorical(data[feature]).codes
    del data['公司全称']
    del data['截止日期_Enddt']
    del data['上市公司代码_Comcd']
    return data


def get_train_test_data(data=None, correlation_threshold=0.7):
    """
    :param correlation_threshold: 默认是0.7
    :param data:
    :return:
    """
    x_data = data.drop("ST是否", axis=1)
    y_data = data.ST是否
    # 处理x_data, 做一个标准化处理
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    # 降维找出相关性过高的列,保存到model中，后续需要用
    features_to_drop = dimensionality_reduction(x_data, y_data, correlation_threshold=correlation_threshold)
    if features_to_drop is None: features_to_drop = []
    # 从数据集中删除高度相关的特征
    x_data_reduced = np.delete(x_data, list(features_to_drop), axis=1)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_data_reduced, y_data, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test, features_to_drop


def get_predict_train_test_data(data=None, features_to_drop=None):
    """
    :param features_to_drop: 需要删除列
    :param data:
    :return:
    """
    x_data = data.drop("ST是否", axis=1)
    y_data = data.ST是否
    # 处理x_data, 做一个标准化处理
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    # 降维找出相关性过高的列,保存到model中，后续需要用
    # features_to_drop = dimensionality_reduction(x_data, y_data, correlation_threshold=0.7)
    if features_to_drop is None: features_to_drop = []
    # 从数据集中删除高度相关的特征
    x_data_reduced = np.delete(x_data, list(features_to_drop), axis=1)
    # 划分训练集和测试集
    return train_test_split(x_data_reduced, y_data, test_size=0.2, random_state=42)
