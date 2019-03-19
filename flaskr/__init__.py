import os
import importlib

from flask import Flask, request
from flaskr.models.base_model import BaseModel
import flaskr.utils as utils
from config import Config as config

available_models = utils.get_available_model_names()


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        print(available_models)
        return 'Hello, World!'

    @app.route('/backtest', methods=['POST'])
    def backtest():
        post_data = request.get_json()
        post_metadata = post_data['metadata']
        # If this is a correct model_name
        if(post_metadata['model_name'] in available_models):
            my_model = BaseModel(
                post_metadata['model_name'], post_metadata['candle_size'], post_metadata['train_daterange'])
            # If there was an existing model, reuse it
            if(my_model.code_name in utils.get_available_exported_model_names()):
                # importlib.import_module('{}.{}'.format(config.EXPORTED_MODELS_MPATH, post_metadata['model_name'])
                return('Cannot reuse model!')
            # Else create new one, train, predict and save it
            else:
                return()

        # Return 404, model_name not matched
        return "this is backtest route: "

    return app
