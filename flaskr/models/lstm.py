import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from flaskr.models.base_model import BaseModel


class Lstm(BaseModel):
    def __init__(self, market_info={}, model_name='', candle_size=1, train_daterange={'from': 0, 'to': 0}):
        BaseModel.__init__(self, market_info, model_name,
                           candle_size, train_daterange)
        self.model = RandomForestClassifier(n_estimators=500)

    def transform_data(self, train_data=None, backtest_data=None):
        x_train = None
        y_train = None
        x_predict = None
        transformed_train_data = []
        transformed_backtest_data = []

        for item in train_data:
            transformed_train_data.append(list(item.values()))

        transformed_train_data = np.array(transformed_train_data)
        # bypass first col (start) and last col(action)
        x_train = transformed_train_data[:, 1:-1]
        y_train = np.reshape(
            transformed_train_data[:, -1], len(transformed_train_data))
        print(transformed_train_data.shape, x_train.shape, y_train.shape)

        for item in backtest_data:
            transformed_backtest_data.append(list(item.values()))
        x_predict = np.array(transformed_backtest_data)[:, 1:]

        print(x_predict.shape)
        return (x_train, y_train, x_predict)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_predict=np.array([])):
        return self.model.predict(x_predict)
