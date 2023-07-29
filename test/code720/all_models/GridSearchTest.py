# @author: code_king
# @date: 2023/7/21 14:57
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from scikeras.wrappers import KerasClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

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
train, test = train_test_split(data, test_size=0.3)

train_X = train.drop('ST是否', axis=1)
train_y = train.ST是否
test_X = test.drop('ST是否', axis=1)
test_y = test.ST是否

# 定义模型
def create_model(dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_X.shape[1], 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 定义参数网格
param_grid = {
    'dropout_rate': [0.1, 0.2, 0.3]
}
# 创建模型对象
model = KerasClassifier(build_fn=create_model, epochs=70, batch_size=32)


# 创建网格搜索对象
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

# 执行网格搜索
grid_search_result = grid_search.fit(train_X.values.reshape(-1, train_X.shape[1], 1), train_y)

# 输出最佳参数和得分
print("Best Parameters: ", grid_search_result.best_params_)
print("Best Score: ", grid_search_result.best_score_)

# 使用最佳参数重新训练模型
best_model = create_model(activation=grid_search_result.best_params_['activation'], dropout_rate=grid_search_result.best_params_['dropout_rate'])
history = best_model.fit(train_X.values.reshape(-1, train_X.shape[1], 1), train_y, epochs=70, batch_size=32, validation_data=(test_X.values.reshape(-1, test_X.shape[1], 1), test_y))

# 预测并评估模型
pred = best_model.predict(test_X.values.reshape(-1, test_X.shape[1], 1))
pred = (pred > 0.5).astype("int32")

print("Accuracy:", metrics.accuracy_score(pred, test_y))
print(classification_report(test_y, pred))

# 绘制训练曲线
plt.plot(history.history_model['loss'])
plt.plot(history.history_model['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history_model['accuracy'])
plt.plot(history.history_model['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# 绘制ROC曲线
pred = best_model.predict(test_X.values.reshape(-1, test_X.shape[1], 1), batch_size=32, verbose=0)
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
