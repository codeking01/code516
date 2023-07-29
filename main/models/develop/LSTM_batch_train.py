# @author: code_king
# @date: 2023/7/27 15:52
from keras import Sequential
from keras.layers import Dense
from keras.src.layers import LSTM, Dropout

from main.models.utils.model_utils import get_data, batch_train, drawing, \
    get_train_test_data


def get_develop_model(train_x=None):
    # 构建LSTM模型
    develop_models = Sequential()
    develop_models.add(LSTM(128, activation='relu', input_shape=(train_x.shape[1], 1)))
    develop_models.add(Dense(256, activation='relu'))
    develop_models.add(Dropout(0.2))
    develop_models.add(Dense(128, activation='relu'))
    develop_models.add(Dropout(0.2))
    develop_models.add(Dense(128, activation='relu'))
    # develop_models.add(Dropout(0.2))
    develop_models.add(Dense(1, activation='sigmoid'))
    develop_models.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return develop_models


# 模型训练及评估
# model.fit(train_X, train_y, epochs=10, batch_size=32, validation_data=(test_X, test_y))
data = get_data("../../data/财务指标36.xlsx")
x_train, x_test, y_train, y_test, features_to_drop = get_train_test_data(data=data, correlation_threshold=0.6)

# 调整输入格式
train_x = x_train.reshape(-1, x_train.shape[1], 1)
test_x = x_test.reshape(-1, x_test.shape[1], 1)
model = get_develop_model(train_x=train_x)
# max_epochs是最大训练轮数 model_name记得修改！
stop_accuracy = 0.76
history_model, y_pred_prob = batch_train(model=model, model_name="LSTM_max_score",
                                         data_list=[train_x, test_x, y_train, y_test], max_epochs=100,
                                         stop_accuracy=stop_accuracy, epochs=300, batch_size=32,
                                         features_to_drop=features_to_drop)
drawing(y_test, y_pred_prob, history_model)
