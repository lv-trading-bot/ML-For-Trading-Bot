import numpy as np
import pandas as pd
import hashlib
import joblib
import json
import requests
import logging
import time
import copy
from sklearn.preprocessing import StandardScaler
from config import Config as config
from flaskr.utils import Utils

logger = logging.getLogger(config.APP_LOGGER_NAME)

MODEL_TYPES = config.MODEL_TYPES
MINUTE_IN_MILLISECONDS = config.MINUTE_IN_MILLISECONDS

ATTRIBUTES_USED_FOR_CODE_NAME = [
    'model_type', 'model_name', 'candle_size', 'market_info', 'lag', 'rolling_step', 'features', 'label']


class BaseModel:
    def __init__(self, model_type=MODEL_TYPES[0], model_name="random_forest", candle_size=60, market_info=None, train_daterange=None, lag=0, rolling_step=0, features=["close", "omlbct"], label="omlbct"):
        self.model_type = model_type
        self.model_name = model_name
        self.candle_size = candle_size
        self.market_info = copy.deepcopy(market_info)
        self.train_daterange = copy.deepcopy(train_daterange)
        self.lag = lag if lag is not None else 0
        self.rolling_step = rolling_step if rolling_step is not None else 0

        self.features = copy.deepcopy(features)
        self.label = label

        self.scaler = StandardScaler()
        self.model = None

        if (model_type == 'rolling' and rolling_step < 1):
            raise Exception('Rolling_step must be > 0')

        if (lag < 0):
            raise Exception('Lag value must be >= 0')

        try:
            horizon = next(x for x in self.features if ('name' in x and x['name'] == self.label))[
                'params']['expirationPeriod']
        except Exception as e:
            logger.error(e)
            raise Exception(
                'Cannot get expirationPeriod in labeled feature\'s params')

        if (train_daterange is None or model_type == 'rolling'):
            logger.info('Using default daterange...')
            now = int(time.time()*1000) - MINUTE_IN_MILLISECONDS

            try:
                train_size = self.train_daterange['to'] - \
                    self.train_daterange['from']
            except:
                train_size = config.DEFAULT_TRAIN_SIZE
                self.train_daterange = {}

            self.train_daterange['to'] = now - \
                (horizon * candle_size * MINUTE_IN_MILLISECONDS)
            self.train_daterange['from'] = self.train_daterange['to'] - train_size

        self.code_name = self.calculate_code_name()

    def calculate_code_name(self):
        all_string_values = []

        if (self.model_type == 'fixed'):
            all_string_values += [str(self.train_daterange['from']), str(self.train_daterange['to'])]
        elif (self.model_type == 'rolling'):
            train_size = self.train_daterange['to'] - self.train_daterange['from']
            all_string_values += [str(train_size)]
        
        for item in ATTRIBUTES_USED_FOR_CODE_NAME:
            all_string_values += Utils.get_string_values_inside(
                getattr(self, item))

        raw_code_name = ''
        for item in all_string_values:
            raw_code_name += str(item) + '$$'
        return hashlib.md5(raw_code_name.encode(encoding='utf-8')).hexdigest()

    def get_candles_by_daterange(self, from_time=0, to_time=0):
        logger.info('Getting candles from %s to %s' % (from_time, to_time))
        return Utils.get_candles_from_db(settings={
            'market_info': self.market_info,
            'candle_size': self.candle_size,
            'from': from_time,
            'to': to_time,
            'features': self.features,
        })

    def get_raw_data(self, data_from, data_to):
        """Get raw data, including pre_data if has lag"""
        pre_data = []
        data = []

        if (self.lag > 0):
            pre_from = data_from - self.lag * self.candle_size * MINUTE_IN_MILLISECONDS
            pre_to = data_from
            pre_data = self.get_candles_by_daterange(pre_from, pre_to)

        data = self.get_candles_by_daterange(data_from, data_to)

        if (len(data) == 0):
            raise Exception('Cannot get data from %s to %s' %
                            (data_from, data_to))
        return pre_data, data

    def turn_into_DataFrame(self, raw_data):
        return pd.DataFrame(raw_data)

    def add_lagged_cols(self, pre_df, data_df, cols_to_drop):
        full_df = pd.concat([pre_df, data_df], ignore_index=True)
        shifted_dfs = [full_df]
        dropped_df = full_df.drop(columns=cols_to_drop)

        if (len(pre_df) == self.lag and self.lag > 0):
            for i in range(1, self.lag + 1):
                shifted_dfs.append(dropped_df.shift(
                    i).add_suffix('_lag%s' % i))
            # drop NaN values
            lagged_df = pd.concat(shifted_dfs, axis=1).dropna()
            return lagged_df
        elif (self.lag == 0):
            return data_df
        else:
            raise Exception('Not enough pre_data to calculate lag')

    def split_x_y(self, data_df, cols_to_drop=[]):
        x = data_df.drop(columns=cols_to_drop).values
        y = data_df[[self.label]].values.reshape(-1)
        return x, y

    def fit_scaler(self, data):
        self.scaler.fit(data)

    def standardize_data(self, data):
        return self.scaler.transform(data)

    def prepare_data(self, raw_pre_data, raw_data, for_training=False):
        # logger.info('Preparing data..., for_training=%s' % for_training)
        pre_df = self.turn_into_DataFrame(raw_pre_data)
        data_df = self.turn_into_DataFrame(raw_data)

        lagged_df = self.add_lagged_cols(
            pre_df, data_df, cols_to_drop=['start', self.label])
        x, y = self.split_x_y(lagged_df, cols_to_drop=['start', self.label])

        if (for_training):
            # logger.info('Fitting new scaler...')
            self.fit_scaler(x)
        x = self.standardize_data(x)

        return x, y

    def train_by_daterange(self, train_from=None, train_to=None):
        if (train_from is None or train_to is None):
            train_from = self.train_daterange['from']
            train_to = self.train_daterange['to']
        else:
            self.train_daterange = {
                'from': train_from,
                'to': train_to
            }

        #update code_name
        self.code_name = self.calculate_code_name()
        
        raw_pre_data, raw_data = self.get_raw_data(train_from, train_to)
        x_train, y_train = self.prepare_data(
            raw_pre_data, raw_data, for_training=True)
        self.train(x_train, y_train)

    def update_by_candle_start(self, candle_start):
        if (self.model_type == 'rolling'):
            candle_size_in_milliseconds = self.candle_size*MINUTE_IN_MILLISECONDS

            # get 'expirationPeriod' attribute in the labeled feature
            horizon = next(x for x in self.features if ('name' in x and x['name'] == self.label))[
                'params']['expirationPeriod']
            # safety gap for getting data to calculate label for train data
            horizon_in_milliseconds = horizon * candle_size_in_milliseconds

            rolling_step_in_milliseconds = self.rolling_step * candle_size_in_milliseconds

            train_to = self.train_daterange['to']
            train_from = self.train_daterange['from']
            train_size = train_to - train_from

            # difference vs first_step in k rolling_step
            diff_vs_first_step = candle_start - \
                (train_to + horizon_in_milliseconds)
            if (diff_vs_first_step >= rolling_step_in_milliseconds):
                logger.info('Rolling to new daterange...')
                block_to_move = (diff_vs_first_step //
                                 rolling_step_in_milliseconds)
                # update daterange and re-train
                self.train_daterange = {
                    'from': train_from + block_to_move * rolling_step_in_milliseconds,
                    'to': train_to + block_to_move * rolling_step_in_milliseconds
                }
                self.train_by_daterange()
                logger.info('Model re-trained!')

    def train(self, x_train, y_train):
        raise NotImplementedError

    def predict(self, x_predict=np.array([])):
        raise NotImplementedError

    def save(self, exported_dir):
        joblib.dump(self, exported_dir + self.code_name + '.joblib')
