import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from flaskr.models.base_model import BaseModel
from config import Config as config

MODEL_TYPES = config.MODEL_TYPES


class RandomForest(BaseModel):
    def __init__(self, model_type=MODEL_TYPES[0], model_name="random_forest", candle_size=60, market_info=None, train_daterange=None, test_daterange=None, lag=0, rolling_step=0):
        BaseModel.__init__(self,
                           model_type=model_type,
                           model_name=model_name,
                           candle_size=candle_size,
                           market_info=market_info,
                           train_daterange=train_daterange,
                           test_daterange=test_daterange,
                           lag=lag,
                           rolling_step=rolling_step)
        self.model = RandomForestClassifier(n_estimators=500)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_predict=np.array([])):
        return self.model.predict(x_predict)
