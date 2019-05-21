class Config:
    APP_LOGGER_NAME = 'flask.app'
    EXPORTED_MODELS_DIR = 'flaskr/exported_models/'
    EXPORTED_MODELS_MPATH = 'flaskr.exported_models'  # module path
    LIVE_MODELS_DIR = 'flaskr/live_models/'
    MODEL_DIR = 'flaskr/models/'

    MODEL_TYPES = ["fixed", "rolling"]
    DB_SERVER_BASE_URL = 'http://localhost:3000'

    DEFAULT_TRAIN_SIZE = 5259487660 # 2 months in millisecond
    DEFAULT_HORIZON = 24
    MINUTE_IN_MILLISECONDS = 60000

    DEFAULT_DROPPED_COLS_WHEN_LAGGING = ['start', 'volume', 'trades', 'hour', 'ADX', 'MACD', 'RSI', 'PLUS_DI', 'MINUS_DI']