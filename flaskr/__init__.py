import os
import joblib
import json
import numpy as np

from flask import Flask, request, abort
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
        return 'Hello, World!'

    # get all available models
    @app.route('/model', methods=['GET'])
    def model():
        return(json.dumps(available_models))

    # backtesting
    @app.route('/backtest', methods=['POST'])
    def backtest():
        post_data = request.get_json()
        post_metadata = post_data['metadata']
        print(post_metadata, available_models)
        # If this is a correct model_name
        if(post_metadata['model_name'] in available_models):
            print('inside')
            my_model = utils.ModelFactory(
                post_metadata['market_info'], post_metadata['model_name'], post_metadata['candle_size'],
                post_metadata['train_daterange'], is_standardized=True, type="rolling", window_size=1)

            x_predict = None
            # # If there was an existing model, reuse it
            # if(my_model.code_name in utils.get_available_exported_model_names()):
            #     print('Using existing model...')
            #     my_model = joblib.load('{}{}.joblib'.format(
            #         config.EXPORTED_MODELS_DIR, my_model.code_name))
            #     x_train, y_train, x_predict = my_model.transform_data(
            #         post_data['train_data'], post_data['backtest_data'])
            # # Else train and save it
            # else:
            print('Creating new model...')
            x_train, y_train, x_predict = my_model.transform_data(
                post_data['train_data'], post_data['backtest_data'])
            my_model.train(x_train, y_train)
            # my_model.save(config.EXPORTED_MODELS_DIR)

            # Finally predict
            if (x_predict.all() != None):
                print('Predicting...')
                y_predict = my_model.predict(x_predict)
                result = {}
                for i in range(len(y_predict)):
                    result['{}'.format(
                        post_data['backtest_data'][i]['start'])] = int(y_predict[i])
                return json.dumps(result)
            else:
                return 'Server cannot predict', 404

        # Return 404, model_name not found
        else:
            return 'model_name not found', 404

    # live trading
    @app.route('/live_trading', methods=['POST'])
    def live_trading():
        return "Live trading"

    return app
