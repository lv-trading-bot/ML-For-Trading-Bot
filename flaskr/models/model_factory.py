import hashlib
import os
import joblib
from flaskr.models.random_forest import RandomForest
from flaskr.models.gradient_boosting import GradientBoosting
from flaskr.models.lstm import Lstm
from config import Config

available_models = {
    "random_forest": RandomForest,
    "gradient_boosting": GradientBoosting,
    "lstm": Lstm
}

MODEL_TYPES = Config.MODEL_TYPES


class ModelFactory:

    def get_available_model_names():
        return list(available_models.keys())

    def model_is_existed(name=""):
        return (name in list(available_models.keys()))

    def create_model(model_type=MODEL_TYPES[0], model_name="random_forest", candle_size=60, market_info=None, train_daterange=None, lag=0, rolling_step=0, features=["close"], label="omlbct"):
        if model_name in available_models:
            try:
                return available_models[model_name](
                    model_type=model_type,
                    model_name=model_name,
                    candle_size=candle_size,
                    market_info=market_info,
                    train_daterange=train_daterange,
                    lag=lag,
                    rolling_step=rolling_step,
                    features=features,
                    label=label)
            except Exception as e:
                raise e
        else:
            return None

    def get_live_models():
        dirs = os.listdir(Config.LIVE_MODELS_DIR)
        result = {}
        for file_name in dirs:
            name = file_name[:-7]  # exclude '.joblib'
            result[name] = joblib.load(Config.LIVE_MODELS_DIR + file_name)
        return result

    # DEPRECATED
    def calculate_code_name(string_array):
        raw_code_name = ''
        sorted_string_array = sorted(string_array)
        for string in sorted_string_array:
            raw_code_name += (string + '$')
        return hashlib.md5(raw_code_name.encode(encoding='utf-8')).hexdigest()
