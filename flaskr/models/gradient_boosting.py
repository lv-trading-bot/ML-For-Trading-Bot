import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from flaskr.models.base_model import BaseModel
from config import Config as config

MODEL_TYPES = config.MODEL_TYPES


class GradientBoosting(BaseModel):
    def __init__(self, model_type=MODEL_TYPES[0], model_name="random_forest", candle_size=60, market_info=None, train_daterange=None, lag=0, rolling_step=0, features=["close", "omlbct"], label="omlbct"):
        BaseModel.__init__(self,
                           model_type=model_type,
                           model_name=model_name,
                           candle_size=candle_size,
                           market_info=market_info,
                           train_daterange=train_daterange,
                           lag=lag,
                           rolling_step=rolling_step,
                           features=features,
                           label=label)
        self.model = GradientBoostingClassifier(
            max_depth=4, learning_rate=0.3, n_estimators=10)

    def prepare_data(self, raw_pre_data, raw_data, for_training=False):
            # logger.info('Preparing data..., for_training=%s' % for_training)
        pre_df = self.turn_into_DataFrame(raw_pre_data)
        data_df = self.turn_into_DataFrame(raw_data)

        lagged_df = self.add_lagged_cols(
            pre_df, data_df, cols_to_drop=([self.label] + config.DEFAULT_DROPPED_COLS_WHEN_LAGGING))

        x, y = self.split_x_y(lagged_df, cols_to_drop=['start', self.label])

        if (for_training):
            # remove last h candles due to OMLBCT strategy
            horizon = self.get_horizon()
            x = x[:-horizon]
            y = y[:-horizon]
            # logger.info('Fitting new scaler...')
            self.fit_scaler(x)
        x = self.standardize_data(x)

        return x, y

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_predict=np.array([])):
        return self.model.predict(x_predict)

    def predict_proba(self, x_predict=np.array([])):
        result = self.model.predict_proba(x_predict)[:, 1]
        return result
