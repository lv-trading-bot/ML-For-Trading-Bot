import os

class Config:
    APP_LOGGER_NAME = 'flask.app'
    PORT = os.getenv('PORT', 3002)
    ID = os.getenv('ID', 'ML_Server_default')
    AUTHENTICATION_TOKEN = os.getenv('AUTHENTICATION_TOKEN')

    EXPORTED_MODELS_DIR = 'flaskr/exported_models/'
    EXPORTED_MODELS_MPATH = 'flaskr.exported_models'  # module path
    LIVE_MODELS_DIR = 'flaskr/live_models/'
    MODEL_DIR = 'flaskr/models/'
    # ID_JSON_FILE = 'flaskr/id.json'

    MODEL_TYPES = ["fixed", "rolling"]
    DB_SERVER_BASE_URL = os.getenv("DB_SERVER_API_BASE", 'http://localhost:3001')

    DEFAULT_TRAIN_SIZE = 5259487660 # 2 months in millisecond
    DEFAULT_HORIZON = 24
    MINUTE_IN_MILLISECONDS = 60000

    DEFAULT_DROPPED_COLS_WHEN_LAGGING = ['start', 'volume', 'trades', 'hour', 'ADX', 'MACD', 'RSI', 'PLUS_DI', 'MINUS_DI']

    SOCKET_URL = os.getenv('SOCKET_URL', 'http://localhost:3004')