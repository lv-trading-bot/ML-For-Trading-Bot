import os
import requests
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


class Utils:

    def get_available_model_names():
        return list(available_models.keys())

    def get_available_exported_model_names():
        dirs = os.listdir(config.EXPORTED_MODELS_DIR)
        result = []
        for file in dirs:
            result.append(file[:-7])  # exclude '.joblib'
        return result

    def ModelFactory(model_type=MODEL_TYPES[0], model_name="random_forest", candle_size=60, market_info=None, train_daterange=None, test_daterange=None, lag=0, rolling_step=0, features=["close"], label="omlbct"):
        if model_name in available_models:
            return available_models[model_name](
                model_type=model_type,
                model_name=model_name,
                candle_size=candle_size,
                market_info=market_info,
                train_daterange=train_daterange,
                test_daterange=test_daterange,
                lag=lag,
                rolling_step=rolling_step,
                features=features,
                label=label)
        else:
            return None

    def filter_dict_array_by_keys(array, keys, not_keys=[]):
        result = []
        for item in array:
            if (not_keys):
                result.append(
                    {k: v for (k, v) in candle.items() if not(k in not_keys)})
            else:
                result.append(
                    {k: v for (k, v) in candle.items() if k in keys})
        return np.array(result)

    def get_candles_from_db(settings):
        """
        {
            "market_info": {
                "exchange": "binance",
                "currency": "USDT",
                "asset": "BTC"
            },
            "candle_size": 60,
            "from": "2018-10-01T00:00:00.000Z",
            "to": "2018-10-01T02:00:00.000Z",
            "features": [
                "start",
                "close",
                "volume",
                "trades",
                {
                        "name": "omlbct",
                        "params": {
                                "takeProfit": 2,
                                "stopLoss": -10,
                                "expirationPeriod": 24
                        }
                }
            ]
        }
        """

        return requests.post(config.DB_SERVER_BASE_URL.join('/candles'), data=settings)
