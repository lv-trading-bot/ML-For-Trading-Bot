import os
from config import Config as config
from flaskr.models.random_forest import RandomForest
from flaskr.models.gradient_boosting import GradientBoosting


def get_available_model_names():
    dirs = os.listdir(config.MODEL_DIR)
    result = []
    for file in dirs:
        if(file.find('.py') != -1 and file != 'base_model.py'):
            result.append(file[:-3])  # exclude '.py'
        else:
            continue
    return result


def get_available_exported_model_names():
    dirs = os.listdir(config.EXPORTED_MODELS_DIR)
    result = []
    for file in dirs:
        result.append(file[:-7])  # exclude '.joblib'
    return result


def ModelFactory(market_info, model_name,  candle_size, train_daterange):
    if(model_name == 'random_forest'):
        return RandomForest(market_info, model_name, candle_size, train_daterange)
    elif(model_name == 'gradient_boosting'):
        return GradientBoosting(market_info, model_name, candle_size, train_daterange)
    else:
        return None
