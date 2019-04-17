import numpy as np
import pandas as pd
import hashlib
import joblib
import json
import requests
import logging
from sklearn.preprocessing import StandardScaler
from config import Config as config
from flaskr.utils import Utils

logger = logging.getLogger(config.APP_LOGGER_NAME)

MODEL_TYPES = config.MODEL_TYPES
MINUTE_IN_MILLISECONDS = 60000


class BaseModel:
    def __init__(self, model_type=MODEL_TYPES[0], model_name="random_forest", candle_size=60, market_info=None, train_daterange=None, test_daterange=None, lag=0, rolling_step=0, features=["close", "omlbct"], label="omlbct"):
        self.model_type = model_type
        self.model_name = model_name
        self.candle_size = candle_size
        self.market_info = market_info
        self.train_daterange = train_daterange
        self.test_daterange = test_daterange
        self.lag = lag if (lag > 0) else 0
        self.rolling_step = rolling_step if (rolling_step > 0) else 0

        self.features = features
        self.label = label
        self.code_name = self.calculate_code_name()
        self.scaler = StandardScaler()
        self.model = None

    def calculate_code_name(self):
        market_info_str = '{}-{}-{}'.format(
            self.market_info['exchange'], self.market_info['currency'], self.market_info['asset'])
        raw_code_name = '{}-{}-{}-{}-{}'.format(market_info_str, self.model_name,
                                                self.candle_size, self.train_daterange['from'], self.train_daterange['to'])
        return hashlib.md5(raw_code_name.encode(encoding='utf-8')).hexdigest()

    def get_candles_by_daterange(self, from_time=0, to_time=0):
        return Utils.get_candles_from_db(settings={
            'market_info': self.market_info,
            'candle_size': self.candle_size,
            'from': from_time,
            'to': to_time,
            'features': self.features,
        })

    def get_raw_data(self):
        """
        Get raw data from DB based on features of model (self.features)

        Returns
        -------
        result : object
            {
                "train" : {
                    "pre_data" : array
                        Prepared data used for calculating lagging
                    "data" : array
                        Actual data used for training
                    "rolling_data" : array
                        Prepared data used for rolling-type model
                },
                "test" : {
                    "pre_data" : array
                        Prepared data used for calculating lagging
                    "data" : array
                        Actual data used for testing
                }

            }
        """

        pre_train = []
        train_data = []
        rolling_train = []

        pre_test = []
        test_data = []

        train_data_from = self.train_daterange['from']
        train_data_to = self.train_daterange['to']
        test_data_from = self.test_daterange['from']
        test_data_to = self.test_daterange['to']

        # pre_train, pre_test used for calculate lag, indicators for train set, test set
        pre_train_from = train_data_from - self.lag * \
            self.candle_size*MINUTE_IN_MILLISECONDS
        pre_train_to = train_data_from
        pre_test_from = test_data_from - self.lag * \
            self.candle_size*MINUTE_IN_MILLISECONDS
        pre_test_to = test_data_from

        # rolling_train used for rolling backtest
        rolling_train_from = train_data_to
        rolling_train_to = train_data_to + (test_data_to - test_data_from)

        # get data from db
        try:
            pre_train = self.get_candles_by_daterange(
                pre_train_from, pre_train_to).json()
            train_data = self.get_candles_by_daterange(
                train_data_from, train_data_to).json()
            rolling_train = self.get_candles_by_daterange(
                rolling_train_from, rolling_train_to).json()

            pre_test = self.get_candles_by_daterange(
                pre_test_from, pre_test_to).json()
            test_data = self.get_candles_by_daterange(
                test_data_from, test_data_to).json()
        except Exception as e:
            return None
        else:
            return {
                "train": {
                    "pre_data": pre_train,
                    "data": train_data,
                    "rolling_data": rolling_train
                },
                "test": {
                    "pre_data": pre_test,
                    "data": test_data,
                }
            }

    def transform_data(self, raw_result):
        """
        Transform data from raw_result to np.ndarray, add lagged columns and standardize data

        Parameters
        ----------
        raw_result : object
            Same structure with the returned value of function get_raw_data

        Returns
        -------
        x_train : np.ndarray
            2D array shaped (observations, features) 
        y_train : np.ndarray
            1D array
        x_rolling : 
            2D array shaped (observations, features)
        y_rolling : np.ndarray
            1D array
        x_predict : np.ndarray
            2D array shaped (observations, features)

        """

        pre_train = pd.DataFrame(raw_result['train']['pre_data'])
        train_data = pd.DataFrame(raw_result['train']['data'])
        rolling_train = pd.DataFrame(raw_result['train']['rolling_data'])

        pre_test = pd.DataFrame(raw_result['test']['pre_data'])
        test_data = pd.DataFrame(raw_result['test']['data'])

        # cols to drop when fitting model
        cols_to_drop = ['start', self.label]

        # If lag>0, add lagged columns to train_data, rolling_train, test_data
        if (self.lag > 0):
            if(self.lag == len(pre_train)):
                # Merge all arrays for easy manipulation
                full_train = pd.DataFrame(
                    raw_result['train']['pre_data'] + raw_result['train']['data'] + raw_result['train']['rolling_data'])
                full_test = pd.DataFrame(
                    raw_result['test']['pre_data'] + raw_result['test']['data'])

                cols_to_concat = [[full_train], [full_test]]

                for i in range(1, self.lag + 1):
                    cols_to_concat[0].append(full_train.drop(
                        columns=cols_to_drop).shift(i).add_suffix('_lag' + str(i)))
                    cols_to_concat[1].append(full_test.drop(
                        columns=cols_to_drop).shift(i).add_suffix('_lag' + str(i)))

                # Drop NaN values
                full_train = pd.concat(cols_to_concat[0], axis=1).dropna()
                full_test = pd.concat(cols_to_concat[1], axis=1).dropna()

                # Split back into train_data, rolling_train
                # test_data is automatically trimmed by dropping NaNs
                train_data = full_train[:len(train_data)]
                rolling_train = full_train[len(train_data):]
                test_data = full_test
            else:
                raise Exception('Non-matching lag value (self.lag, len(pre_train)) = ({}, {})'.format(
                    self.lag, len(pre_train)))

        logger.info('Train set[:5]: \n{}'.format(train_data.head()))
        logger.info('Test set[:5]: \n{}'.format(test_data.head()))

        # filter out cols
        x_train = train_data.drop(columns=cols_to_drop).values
        y_train = train_data[[self.label]].values.reshape(-1)
        x_rolling = rolling_train.drop(columns=cols_to_drop).values
        y_rolling = rolling_train[[self.label]].values.reshape(-1)
        x_predict = test_data.drop(columns=cols_to_drop).values

        # standardize data
        if (self.scaler):
            self.scaler.fit(x_train)
            x_train = self.scaler.transform(x_train)
            x_rolling = self.scaler.transform(x_rolling)
            x_predict = self.scaler.transform(x_predict)

        return x_train, y_train, x_rolling, y_rolling, x_predict

    def train(self, x_train, y_train):
        raise NotImplementedError

    def predict(self, x_predict=np.array([])):
        raise NotImplementedError

    def save(self, exported_dir):
        joblib.dump(self, exported_dir + self.code_name + '.joblib')
