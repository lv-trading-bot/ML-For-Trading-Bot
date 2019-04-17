class Config:
    APP_LOGGER_NAME = 'flask.app'
    EXPORTED_MODELS_DIR = 'flaskr/exported_models/'
    EXPORTED_MODELS_MPATH = 'flaskr.exported_models'  # module path
    MODEL_DIR = 'flaskr/models/'
    MODEL_TYPES = ["fixed", "rolling"]
    DB_SERVER_BASE_URL = 'http://localhost:3000'