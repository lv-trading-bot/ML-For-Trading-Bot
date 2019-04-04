import numpy as np
import hashlib
import joblib
from sklearn.preprocessing import StandardScaler


class BaseModel:
    def __init__(self, market_info={}, model_name='', candle_size=1, train_daterange={'from': 0, 'to': 0},
                 is_standardized=True, method="default", rolling_step=0):
        self.market_info = market_info
        self.model_name = model_name
        self.candle_size = candle_size
        self.train_daterange = train_daterange
        self.code_name = self.calculate_code_name()
        self.is_standardized = is_standardized
        self.method = method
        self.rolling_step = rolling_step

        self.scaler = StandardScaler() if self.is_standardized else None
        self.model = None

    def calculate_code_name(self):
        market_info_str = '{}-{}-{}'.format(
            self.market_info['exchange'], self.market_info['currency'], self.market_info['asset'])
        raw_code_name = '{}-{}-{}-{}-{}'.format(market_info_str, self.model_name,
                                                self.candle_size, self.train_daterange['from'], self.train_daterange['to'])
        return hashlib.md5(raw_code_name.encode(encoding='utf-8')).hexdigest()

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
