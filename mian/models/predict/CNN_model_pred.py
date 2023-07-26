# @author: code_king
# @date: 2023/7/25 22:13

import joblib

from mian.models.utils.model_utils import get_data, get_train_test_data
from mian.models.utils.predict_utils import get_score

# 加载模型和降维删除的列
models = joblib.load('../save_models/models.joblib')
model = models['model']
features_to_drop = models['features_to_drop']

data = get_data("../../data/财务指标36.xlsx")
x_train, x_test, y_train, y_test = get_train_test_data(data=data)

get_score(model, x_train, x_test, y_train, y_test)
