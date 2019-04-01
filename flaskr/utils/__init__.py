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

def get_available_model_names():
    return list(available_models.keys())

def get_available_exported_model_names():
    dirs = os.listdir(config.EXPORTED_MODELS_DIR)
    result = []
    for file in dirs:
        result.append(file[:-7])  # exclude '.joblib'
    return result


def ModelFactory(market_info, model_name,  candle_size, train_daterange):
    if model_name in available_models:
        return available_models[model_name](market_info, model_name, candle_size, train_daterange)
    else:
        return None
