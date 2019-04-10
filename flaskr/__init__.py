import os
import joblib
import json
import numpy as np
import math
import time

from flask import Flask, request, abort, g
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

    @app.before_request
    def before_request():
        g.request_start_time = time.time()

    @app.after_request
    def after_request(response):
        app.logger.info('Execution time %ss', round(
            time.time() - g.request_start_time, 5))
        return response

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
        app.logger.info('POST metada: %s', post_metadata)
        # If this is a correct model_name
        if(post_metadata['model_name'] in available_models):
            app.logger.info('Creating new model...')
            my_model = utils.ModelFactory2(
                model_type=post_metadata['model_type'],
                model_name=post_metadata['model_name'],
                candle_size=post_metadata['candle_size'],
                market_info=post_metadata['market_info'],
                train_daterange=post_metadata['train_daterange'],
                test_daterange=post_data['test_daterange'],
                lag=post_metadata['lag'],
                rolling_step=post_metadata['rolling_step'])

            # x_predict = None
            # x_train, y_train, x_predict = my_model.transform_data(
            #     post_data['train_data'], post_data['backtest_data'])
            # y_predict = np.array([])
            # if (my_model.method == 'default'):
            #     my_model.train(x_train, y_train)
            #     y_predict = my_model.predict(x_predict)

            # elif (my_model.method == 'rolling'):
            #     if (my_model.rolling_step < 1):
            #         return 'Invalid rolling step', 400
            #     while (len(x_predict) != 0):
            #         print(len(x_predict))
            #         my_model.train(x_train, y_train)
            #         new_predictions = my_model.predict(
            #             x_predict[:my_model.rolling_step])
            #         actual_predictions_length = len(new_predictions)
            #         y_predict = np.append(y_predict, new_predictions)
            #         # perform sliding window:
            #         # append new predictions to x_train, y_train, also remove old ones
            #         x_train = np.append(
            #             x_train[actual_predictions_length:], x_predict[:actual_predictions_length], axis=0)
            #         y_train = np.append(
            #             y_train[actual_predictions_length:], new_predictions)
            #         # shift and cut predicted rows
            #         x_predict = x_predict[actual_predictions_length:]
            # else:
            #     return 'Unknown backtest method', 404

            app.logger.info('Predicting...')

            # Send result
            result = {}
            for i in range(len(y_predict)):
                result['{}'.format(
                    post_data['backtest_data'][i]['start'])] = int(y_predict[i])
            return json.dumps(result)

        # Return 404, model_name not found
        else:
            return 'model_name not found', 404

    # live trading
    @app.route('/live_trading', methods=['POST'])
    def live_trading():
        return "Live trading"

    return app
