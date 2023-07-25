# @author: code_king
# @date: 2023/7/21 14:10
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics

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
from sklearn.model_selection import train_test_split, GridSearchCV

train, test = train_test_split(data, test_size=0.3)

train_X = train.drop('ST是否', axis=1)
train_y = train.ST是否
test_X = test.drop('ST是否', axis=1)
test_y = test.ST是否

# 定义要调整的超参数范围
lstm_units = [32, 64, 128]
dropout = [0.1, 0.2, 0.3]
optimizers = ['adam', 'rmsprop']


# 定义Keras模型
def build_model(lstm_units=64, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=(train_X.shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 转换为Sci-Keras格式
# 可以适当增加 epochs的次数
model = KerasClassifier(build_fn=build_model, epochs=10, batch_size=32)

# 网格搜索
grid = GridSearchCV(model, param_grid=dict(lstm_units=lstm_units, dropout=dropout), cv=3)

grid_result = grid.fit(train_X, train_y)

# 汇总结果
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# 使用最优参数预测和评估
best_model = grid.best_estimator_
pred = best_model.predict(test_X)
print("Accuracy:", metrics.accuracy_score(pred, test_y))
