# @author: code_king
# @date: 2023/7/26 20:27
import numpy as np
from sklearn.metrics import accuracy_score


def get_score(model, x_train, x_test, y_train, y_test):
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
    y_pred_data = model.predict(np.r_[x_train, x_test])
    # 模型评估
    pred = (y_pred_data > 0.5).astype("int32")
    accuracy = accuracy_score(pred, (np.r_[y_train, y_test]))
    print("整体数据 accuracy:", accuracy)
