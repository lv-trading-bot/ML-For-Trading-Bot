import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from flaskr.models.base_model import BaseModel


class RandomForest(BaseModel):
    def __init__(self, model_name='', candle_size=1, train_daterange={'from': 0, 'to': 0}):
        BaseModel.__init__(self, model_name, candle_size, train_daterange)
        self.model = RandomForestClassifier(n_estimators=500)

    def transform_data(self, train_data=None, backtest_data=None):
        x_train = None
        y_train = None
        x_predict = None
        # TODO Implement transformation
        return (x_train, y_train, x_predict)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_predict=np.array([])):
        return self.model.predict(x_predict)
