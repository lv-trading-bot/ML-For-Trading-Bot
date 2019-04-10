import numpy as np
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

        self.code_name = self.calculate_code_name()
        self.scaler = StandardScaler()
        self.model = None

    def calculate_code_name(self):
        market_info_str = '{}-{}-{}'.format(
            self.market_info['exchange'], self.market_info['currency'], self.market_info['asset'])
        raw_code_name = '{}-{}-{}-{}-{}'.format(market_info_str, self.model_name,
                                                self.candle_size, self.train_daterange['from'], self.train_daterange['to'])
        return hashlib.md5(raw_code_name.encode(encoding='utf-8')).hexdigest()

    # get data from DB, right now temporarily read from JSON
    def get_raw_data(self):
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
                "pre_data": np.array(pre_train),
                "data": np.array(train_data),
                "rolling_data": np.array(rolling_train)
            },
            "test": {
                "pre_data": np.array(pre_test),
                "data": np.array(test_data),
            }
        }

    def transform_data(self, train_data=None, backtest_data=None):
        x_train = None
        y_train = None
        x_predict = None
        transformed_train_data = []
        transformed_backtest_data = []

        for item in train_data:
            transformed_train_data.append(list(item.values()))
        transformed_train_data = np.array(transformed_train_data)

        for item in backtest_data:
            transformed_backtest_data.append(list(item.values()))
        transformed_backtest_data = np.array(transformed_backtest_data)

        # bypass first col (start) and last col(action)
        x_train = transformed_train_data[:, 1:-1]
        y_train = np.reshape(
            transformed_train_data[:, -1], len(transformed_train_data))
        print(transformed_train_data.shape, x_train.shape, y_train.shape)

        x_predict = transformed_backtest_data[:, 1:]
        print(x_predict.shape)

        if (self.is_standardized):
            self.scaler.fit(x_train)
            x_train = self.scaler.transform(x_train)
            x_predict = self.scaler.transform(x_predict)

        return (x_train, y_train, x_predict)

    def train(self, x_train, y_train):
        raise NotImplementedError

    def predict(self, x_predict=np.array([])):
        raise NotImplementedError

    def save(self, exported_dir):
        joblib.dump(self, exported_dir + self.code_name + '.joblib')


# # Check code
# bmodel = BaseModel(market_info={
#     "exchange": "binance",
#     "asset": "BTC",
#     "currency": "USDT"
# },
#     train_daterange={
#     "from": 1518382800000,
#     "to": 1522404000000
# },
#     test_daterange={
#     "from": 1522404000000,
#     "to": 1522422000000
# },
#     lag=1,
#     rolling_step=2)

# result = bmodel.get_raw_data()

# print("TRAIN\n", result['train']['pre_data'])
# print(result['train']['data'].shape, "\n", result['train']
#       ['data'][0], result['train']['data'][-1])
# print(result['train']['rolling_data'].shape, "\n", result['train']
#       ['rolling_data'][0], result['train']['rolling_data'][-1])

# print("TEST\n", result['test']['pre_data'])
# print(result['test']['data'].shape, "\n", result['test']
#       ['data'][0], result['test']['data'][-1])
