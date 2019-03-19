import os
from config import Config as config


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
