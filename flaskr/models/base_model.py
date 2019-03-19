import numpy as np
import hashlib
import joblib


class BaseModel:
    def __init__(self, model_name='', candle_size=1, train_daterange={'from': 0, 'to': 0}):
        self.model_name = model_name
        self.candle_size = candle_size
        self.train_daterange = train_daterange
        self.code_name = self.calculate_code_name()
        self.model = None

    def calculate_code_name(self):
        raw_code_name = '{}-{}-{}-{}'.format(self.model_name, self.candle_size,
                                             self.train_daterange['from'], self.train_daterange['to'])
        return hashlib.md5(raw_code_name.encode(encoding='utf-8')).hexdigest()

    def transform_data(self, train_data=None, backtest_data=None):
        x_train = None
        y_train = None
        x_predict = None
        return (x_train, y_train, x_predict)

    def train(self, x_train, y_train):
        raise NotImplementedError

    def predict(self, x_predict=np.array([])):
        raise NotImplementedError

    def save(self, exported_dir):
        joblib.dump(self, exported_dir + self.code_name + '.joblib')
