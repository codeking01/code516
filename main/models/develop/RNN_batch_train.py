# @author: code_king
# @date: 2023/7/27 15:55
from keras import Sequential
from keras.layers import Dense
from keras.src.layers import Dropout, SimpleRNN, Flatten

from main.models.utils.model_utils import get_data, batch_train, drawing, \
    get_train_test_data


def get_develop_model(train_x=None):
    model = Sequential()
    model.add(SimpleRNN(units=128, return_sequences=True, input_shape=(train_x.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=100))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 模型训练及评估
# model.fit(train_X, train_y, epochs=10, batch_size=32, validation_data=(test_X, test_y))
data = get_data("../../data/财务指标36.xlsx")
x_train, x_test, y_train, y_test, features_to_drop = get_train_test_data(data=data, correlation_threshold=0.7)

# 调整输入格式
train_x = x_train.reshape(-1, x_train.shape[1], 1)
test_x = x_test.reshape(-1, x_test.shape[1], 1)

model = get_develop_model(train_x=train_x)
# max_epochs是最大训练轮数 model_name记得修改！
stop_accuracy = 0.8
history_model, y_pred_prob = batch_train(model=model, model_name="RNN_max_score",
                                         data_list=[train_x, test_x, y_train, y_test], max_epochs=1000,
                                         stop_accuracy=stop_accuracy, epochs=30, batch_size=32,
                                         features_to_drop=features_to_drop)
drawing(y_test, y_pred_prob, history_model)
