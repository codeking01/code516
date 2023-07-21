# @author: code_king
# @date: 2023/7/20 20:29
# 导入必要的包
import warnings

import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")
# 忽略警告
import matplotlib.pyplot as plt

data = pd.read_excel('../源文件/财务数据11(3).xlsx')
# 读取数据
data['ST是否'] = data['ST是否'].fillna('1')
data['截止日期_Enddt'] = str(data['截止日期_Enddt'])
# 把时间数据转换为字符串，这样模型才能处理

data = data.dropna()
# 删除空值

for feature in data.columns:
    if data[feature].dtype == 'object':
        data[feature] = pd.Categorical(data[feature]).codes
# 如果数据类型是object就把她转换成分类数据
del data['公司全称']
del data['截止日期_Enddt']
del data['上市公司代码_Comcd']
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.3)
# 按照7：3划分训练集和验证集
train_X = train.drop('ST是否', axis=1)
# 训练集x
train_y = train.ST是否
# 训练集y
test_X = test.drop('ST是否', axis=1)
# 测试集x
test_y = test.ST是否
# 测试集y

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
# 逻辑回归模型
model.fit(train_X, train_y)
# 训练模型
prediction = model.predict(test_X)
# 使用模型进行预测
print('acc：', metrics.accuracy_score(prediction, test_y))
# 输出acc
print(classification_report(test_y, prediction, target_names=['0', '1']))
# 输出分类报告
pred = model.predict_proba(test_X)[:, 1]
# 获得预测值
fpr, tpr, threshold = metrics.roc_curve(test_y, pred)
# 获得fpr和tpr
roc_auc = metrics.auc(fpr, tpr)
# 绘画roc——auc

# 随机森林
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('acc：', metrics.accuracy_score(prediction, test_y))
print(classification_report(test_y, prediction, target_names=['0', '1']))
pred = model.predict_proba(test_X)[:, 1]
fpr1, tpr1, threshold1 = metrics.roc_curve(test_y, pred)
roc_auc1 = metrics.auc(fpr1, tpr1)

plt.figure(figsize=(6, 6))
plt.title('Validation ROC')
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.plot(fpr, tpr, 'r', label='Logistic AUC = %0.3f' % roc_auc)
# plt.plot(fpr1, tpr1, 'b', label = 'RS AUC = %0.3f' % roc_auc1)
# plt.plot(fpr2, tpr2, label = 'NN AUC = %0.3f' % roc_auc2)
# plt.plot(fpr3, tpr3,  label = 'XBG AUC = %0.3f' % roc_auc3)
# plt.plot(fpr4, tpr4,  label = 'SVM AUC = %0.3f' % roc_auc4)
plt.plot([0, 1], [0, 1], 'r--')
plt.legend()
plt.show()
