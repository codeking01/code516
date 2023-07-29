# @author: code_king
# @date: 2023/7/25 22:13

import joblib

from main.models.utils.model_utils import get_data, get_predict_train_test_data, drawing
from main.models.utils.predict_utils import get_score


def get_predict_drawing(model_name=None, data_path=None):
    # 加载模型和降维删除的列
    models = joblib.load(f'{model_name}')
    history_model = models['model']
    predict_model = history_model.model
    features_to_drop = models['features_to_drop']
    data = get_data(f"{data_path}")
    x_train, x_test, y_train, y_test = get_predict_train_test_data(data=data, features_to_drop=features_to_drop)
    get_score(predict_model, model_name, x_train, x_test, y_train, y_test)
    # 这个只画了测试集的图片
    drawing(y_test=y_test, y_pred_prob=predict_model.predict(x_test), history=history_model)


# 根据选择的模型去看相应的结果
get_predict_drawing(model_name="../save_models/CNN_max_score.joblib", data_path="../../data/财务指标36.xlsx")
get_predict_drawing(model_name="../save_models/GCN_max_score.joblib", data_path="../../data/财务指标36.xlsx")
get_predict_drawing(model_name="../save_models/LSTM_max_score.joblib", data_path="../../data/财务指标36.xlsx")
get_predict_drawing(model_name="../save_models/RNN_max_score.joblib", data_path="../../data/财务指标36.xlsx")
get_predict_drawing(model_name="../save_models/GRU_max_score.joblib", data_path="../../data/财务指标36.xlsx")
get_predict_drawing(model_name="../save_models/BPNN_max_score.joblib", data_path="../../data/财务指标36.xlsx")
