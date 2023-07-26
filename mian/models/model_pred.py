# @author: code_king
# @date: 2023/7/25 22:13

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBModel

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

model = joblib.load(filename="model.joblib")
y_pred = model.predict(x_test)
# 模型评估
pred = (y_pred > 0.5).astype("int32")
accuracy = accuracy_score(pred, y_test)
print("测试集 accuracy:", accuracy)

#  训练集
y_pred_train = model.predict(x_train)
# 模型评估
pred = (y_pred_train > 0.5).astype("int32")
accuracy = accuracy_score(pred, y_train)
print("训练集 accuracy:", accuracy)

#  训练集
y_pred_data = model.predict(np.r_[x_train,x_test])
# 模型评估
pred = (y_pred_data > 0.5).astype("int32")
accuracy = accuracy_score(pred, (np.r_[y_train,y_test]))
print("整体数据 accuracy:", accuracy)