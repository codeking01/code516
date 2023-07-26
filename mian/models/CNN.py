# @author: code_king
# @date: 2023/7/20 21:00
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('../data/财务指标36.xlsx')

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

x_data = data.drop("ST是否", axis=1)
y_data = data.ST是否

# 处理x_data, 做一个标准化处理
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

# 计算特征之间的相关性矩阵
corr_matrix = np.abs(np.corrcoef(x_data, rowvar=False))
# 设置相关性的阈值，根据需要进行调整
correlation_threshold = 0.7
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

# 从数据集中删除高度相关的特征
x_data_reduced = np.delete(x_data, list(features_to_drop), axis=1)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x_data_reduced, y_data, test_size=0.2, random_state=42)

# 调整输入格式
train_X = x_train.reshape(-1, x_train.shape[1], 1)
test_X = x_test.reshape(-1, x_test.shape[1], 1)

def develop_model():
    global model
    model = Sequential()
    model.add(Conv1D(256, 3, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

develop_model()

# 定义停止条件
max_epochs = 500  # 最大训练轮数
stop_accuracy = 0.81  # 停止训练的准确率阈值

# 开始训练的死循环
for epoch in range(max_epochs):
    history = model.fit(train_X, y_train, epochs=1, batch_size=32, validation_data=(test_X, y_test), verbose=0)

    # 模型评估
    pred = model.predict(test_X)
    pred = (pred > 0.5).astype("int32")
    accuracy = metrics.accuracy_score(pred, y_test)
    print("Epoch: {} - Accuracy: {}".format(epoch + 1, accuracy))

    # 当准确率超过阈值时保存模型
    if accuracy > stop_accuracy:
        joblib.dump(model, 'model.joblib')
        print("Model saved.")
        break

# 使用模型预测概率值，而不是类别标签
y_pred_prob = model.predict(test_X)[:, 0]

# 计算ROC曲线
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

# 计算AUC值
roc_auc = metrics.auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(6, 6))
plt.title('Validation ROC')
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.plot(fpr, tpr, 'r', label='CNN AUC = %0.3f' % roc_auc)
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



# # 绘制ROC曲线
# fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
# roc_auc = metrics.auc(fpr, tpr)
# plt.figure(figsize=(6, 6))
# plt.title('Validation ROC')
# plt.xlabel('False Positive Rate', fontsize=13)
# plt.ylabel('True Positive Rate', fontsize=13)
# plt.plot(fpr, tpr, 'r', label='CNN AUC = %0.3f' % roc_auc)
# plt.plot([0, 1], [0, 1], 'r--')
# plt.legend()
# plt.show()


# 定义要调整的超参数的网格
# from sklearn.model_selection import GridSearchCV
# from keras.wrappers.scikit_learn import KerasClassifier
# param_grid = {
#     'epochs': [10, 20],
#     'batch_size': [32, 64],
#     'optimizer': ['adam', 'rmsprop']
# }
#
# # 使用GridSearchCV对超参数进行网格搜索
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
# grid_result = grid.fit(x_train, y_train)
#
# # 打印最佳参数和得分
# print("Best parameters: ", grid_result.best_params_)
# # print("Best score: ", grid_result.best_score_)
