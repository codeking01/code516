# @author: code_king
# @date: 2023/7/25 21:00
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

from main.models.utils.model_utils import get_data, batch_train, drawing, get_train_test_data


def get_develop_model(x_train=None):
    develop_models = Sequential()
    develop_models.add(Conv1D(256, 3, activation='relu', input_shape=(x_train.shape[1], 1)))
    develop_models.add(MaxPooling1D(pool_size=2))
    develop_models.add(Conv1D(128, 3, activation='relu'))
    develop_models.add(MaxPooling1D(pool_size=2))
    develop_models.add(Flatten())
    develop_models.add(Dense(512, activation='relu'))
    develop_models.add(Dropout(0.2))
    develop_models.add(Dense(256, activation='relu'))
    develop_models.add(Dropout(0.2))
    develop_models.add(Dense(1, activation='sigmoid'))
    # sigmoid激活函数的范围是0到1之间的连续值,表示该样本属于正类的概率。
    develop_models.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return develop_models


data = get_data("../../data/财务指标36.xlsx")
x_train, x_test, y_train, y_test, features_to_drop = get_train_test_data(data=data)

# 调整输入格式
train_x = x_train.reshape(-1, x_train.shape[1], 1)
test_x = x_test.reshape(-1, x_test.shape[1], 1)

model = get_develop_model(x_train=train_x)
# max_epochs是最大训练轮数
stop_accuracy = 0.8
history_model, y_pred_prob = batch_train(model=model, model_name="CNN_max_score",
                                         data_list=[train_x, test_x, y_train, y_test], max_epochs=1000,
                                         stop_accuracy=stop_accuracy, epochs=30, batch_size=32,
                                         features_to_drop=features_to_drop)
drawing(y_test, y_pred_prob, history_model)

# 最高分： 测试集 accuracy: 0.8516746411483254
