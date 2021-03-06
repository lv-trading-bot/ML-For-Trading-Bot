import os
import requests
import logging
import json
import joblib
from config import Config as config

logger = logging.getLogger(config.APP_LOGGER_NAME)


class Utils:

    def get_available_exported_model_names():
        dirs = os.listdir(config.EXPORTED_MODELS_DIR)
        result = []
        for file in dirs:
            result.append(file[:-7])  # exclude '.joblib'
        return result

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
        """Get raw candles from DB

        Parameters
        ----------
        settings : object
        {
            "market_info": {
                "exchange": "binance",
                "currency": "USDT",
                "asset": "BTC"
            },
            "candle_size": 60,
            "from": 1555925760000,
            "to": 1555929360000,
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

        Returns
        -------
        candles: array
            Return array of dict if success, otherwise return None
        """
        try:
            return requests.post(config.DB_SERVER_BASE_URL + '/candles', json=settings).json()
        except Exception as e:
            raise e
            return []

    def get_string_values_inside(obj):
        result = []
        
        if (type(obj) is dict):
            for item in obj:
                result += Utils.get_string_values_inside(obj[item])
        elif (type(obj) is list):   
            for item in obj:
                result += Utils.get_string_values_inside(item)    
        else:
            result.append(str(obj))

        return sorted(result)
