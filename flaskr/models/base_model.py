import numpy as np
import pandas as pd
import hashlib
import joblib
import json
from sklearn.preprocessing import StandardScaler
from config import Config as config

MODEL_TYPES = config.MODEL_TYPES
MINUTE_IN_MILLISECONDS = 60000


class BaseModel:
    def __init__(self, model_type=MODEL_TYPES[0], model_name="random_forest", candle_size=60, market_info=None, train_daterange=None, test_daterange=None, lag=0, rolling_step=0):
        self.model_type = model_type
        self.model_name = model_name
        self.candle_size = candle_size
        self.market_info = market_info
        self.train_daterange = train_daterange
        self.test_daterange = test_daterange
        self.lag = lag if (lag > 0) else 0
        self.rolling_step = rolling_step if (rolling_step > 0) else 0

        self.features = ["start", "open", "high", "low",
                         "close", "volume", "trades", "action"]
        self.code_name = self.calculate_code_name()
        self.scaler = StandardScaler()
        self.model = None

    def calculate_code_name(self):
        market_info_str = '{}-{}-{}'.format(
            self.market_info['exchange'], self.market_info['currency'], self.market_info['asset'])
        raw_code_name = '{}-{}-{}-{}-{}'.format(market_info_str, self.model_name,
                                                self.candle_size, self.train_daterange['from'], self.train_daterange['to'])
        return hashlib.md5(raw_code_name.encode(encoding='utf-8')).hexdigest()

    def get_raw_data(self, features=["start", "open", "high", "low", "close", "volume", "trades", "action"]):
        """
        Get data from DB by features, right now temporarily read from JSON
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

        # Load JSON, assume data is in ascending order
        with open('flaskr/data/full_{}_{}_{}_OMLBCT_{}_01-09-17_25-02-19.json'.format(
            self.market_info['exchange'], self.market_info['asset'], self.market_info['currency'], self.candle_size
        )) as json_file:
            candles = json.load(json_file)

            # get data by daterange
            for candle in candles:
                candle_start = candle['start']
                # filter out attributes
                candle = {k: v for (k, v) in candle.items() if k in features}
                # if candle falls into any range of the pre_train, train, ...
                if (candle_start >= pre_train_from and candle_start < pre_train_to):
                    pre_train.append(candle)
                if (candle_start >= train_data_from and candle_start < train_data_to):
                    train_data.append(candle)
                if (candle_start >= rolling_train_from and candle_start < rolling_train_to):
                    rolling_train.append(candle)
                if (candle_start >= pre_test_from and candle_start < pre_test_to):
                    pre_test.append(candle)
                if (candle_start >= test_data_from and candle_start < test_data_to):
                    test_data.append(candle)

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
        Transform data from raw_result to np.array, add lagged columns and standardize data
        """

        pre_train = pd.DataFrame(raw_result['train']['pre_data'])
        train_data = pd.DataFrame(raw_result['train']['data'])
        rolling_train = pd.DataFrame(raw_result['train']['rolling_data'])
        print(len(pre_train), len(train_data), len(rolling_train))

        pre_test = pd.DataFrame(raw_result['test']['pre_data'])
        test_data = pd.DataFrame(raw_result['test']['data'])

        # If lag>0, add lagged columns to train_data, rolling_train, test_data
        if (self.lag > 0):
            if(self.lag == len(pre_train)):
                # Merge all arrays for easy manipulation
                full_train = pd.DataFrame(
                    raw_result['train']['pre_data'] + raw_result['train']['data'] + raw_result['train']['rolling_data'])
                full_test = pd.DataFrame(
                    raw_result['test']['pre_data'] + raw_result['test']['data'])

                cols_to_drop = ['start', 'action']
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
                print(train_data.shape, rolling_train.shape, full_test.shape)
            else:
                print('Lag value is invalid.')

        print(train_data.head())
        print(test_data.head())

        # filter out cols
        x_train = train_data.drop(columns=['start', 'action']).values
        y_train = train_data[['action']].values.reshape(-1)
        x_rolling = rolling_train.drop(columns=['start', 'action']).values
        y_rolling = rolling_train[['action']].values.reshape(-1)
        x_predict = test_data.drop(columns=['start', 'action']).values

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
