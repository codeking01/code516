# @author: code_king
# @date: 2023/7/20 21:00
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 读取和预处理数据
data = pd.read_excel('../源文件/财务数据11(3).xlsx')
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

# 构建LSTM模型
model = Sequential()
model.add(LSTM(64, input_shape=(train_X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 调整输入格式
train_X = train_X.values.reshape(-1, train_X.shape[1], 1)
test_X = test_X.values.reshape(-1, test_X.shape[1], 1)

# 模型训练及评估
history = model.fit(train_X, train_y, epochs=70, batch_size=32, validation_data=(test_X, test_y))

pred = model.predict(test_X)
pred = (pred > 0.5).astype("int32")

print("Accuracy:", metrics.accuracy_score(pred, test_y))
print(classification_report(test_y, pred))

# 绘制训练曲线
# history = model.fit(train_X, train_y, epochs=70, batch_size=32, validation_data=(test_X, test_y))

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

pred = model.predict(test_X, batch_size=32, verbose=0)
# 绘制ROC曲线
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.title('Validation ROC')
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.plot(fpr, tpr, 'r', label='Logistic AUC = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.legend()
plt.show()
