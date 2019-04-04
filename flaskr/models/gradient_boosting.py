import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from flaskr.models.base_model import BaseModel


class GradientBoosting(BaseModel):
    def __init__(self, market_info={}, model_name='', candle_size=1, train_daterange={'from': 0, 'to': 0},
                 is_standardized=True, type="default", window_size=1):
        BaseModel.__init__(self, market_info, model_name,
                           candle_size, train_daterange, is_standardized, type, window_size)
        self.model = GradientBoostingClassifier(
            max_depth=4, learning_rate=0.3, n_estimators=10)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_predict=np.array([])):
        return self.model.predict(x_predict)
