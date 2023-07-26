import matplotlib.pyplot as plt
import pandas as pd
from keras import Sequential
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
# 读取和预处理数据
data = pd.read_excel('../data/财务指标36.xlsx')
data['ST是否'] = data['ST是否'].fillna('1')
data['截止日期_Enddt'] = str(data['截止日期_Enddt'])
data = data.dropna()

for feature in data.columns:
    if data[feature].dtype == 'object':
        data[feature] = pd.Categorical(data[feature]).codes

del data['公司全称']
del data['截止日期_Enddt']
del data['上市公司代码_Comcd']

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.3)

train_X = train.drop('ST是否', axis=1)
train_y = train.ST是否
test_X = test.drop('ST是否', axis=1)
test_y = test.ST是否

# 构建BPNN模型
model = Sequential()
model = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=10000)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 调整输入格式
train_X = train_X.values.reshape(-1, train_X.shape[1])
test_X = test_X.values.reshape(-1, test_X.shape[1])
# x_train_2D = (x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
# x_test_2D = (x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
# 模型训练及评估
model.fit(train_X, train_y)

pred = model.predict(test_X)
pred = (pred > 0.5).astype("int32")

print("Accuracy:", metrics.accuracy_score(pred, test_y))
print(classification_report(test_y, pred))

pred = model.predict(test_X)
# 绘制ROC曲线
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.title('Validation ROC')
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.plot(fpr, tpr, 'r', label='BPNN AUC = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.legend()
plt.show()
# # 绘制训练曲线
# history = model.fit(train_X, train_y)
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='lower right')
# plt.show()
