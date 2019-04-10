import os
from config import Config as config
from flaskr.models.random_forest import RandomForest
from flaskr.models.gradient_boosting import GradientBoosting
from flaskr.models.lstm import Lstm

available_models = {
    "random_forest": RandomForest,
    "gradient_boosting": GradientBoosting,
    "lstm": Lstm
}

MODEL_TYPES = config.MODEL_TYPES


def get_available_model_names():
    return list(available_models.keys())


def get_available_exported_model_names():
    dirs = os.listdir(config.EXPORTED_MODELS_DIR)
    result = []
    for file in dirs:
        result.append(file[:-7])  # exclude '.joblib'
    return result


def ModelFactory(market_info, model_name,  candle_size, train_daterange, is_standardized, method, rolling_step):
    if model_name in available_models:
        return available_models[model_name](market_info, model_name, candle_size, train_daterange, is_standardized, method, rolling_step)
    else:
        return None


def ModelFactory2(model_type=MODEL_TYPES[0], model_name="random_forest", candle_size=60, market_info=None, train_daterage=None, test_daterange=None, lag=0, rolling_step=0):
    if model_name in available_models:
        return available_models[model_name](
            model_type=model_type,
            model_name=model_name,
            candle_size=candle_size,
            market_info=market_info,
            train_daterage=train_daterage,
            test_daterange=test_daterange,
            lag=lag,
            rolling_step=rolling_step)
    else:
        return None
