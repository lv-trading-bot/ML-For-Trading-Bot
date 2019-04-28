import os
import joblib
import json
import numpy as np
import math
import time
import pandas as pd

from flask import Flask, request, abort, g
from flaskr.utils import Utils as utils
from flaskr.models.model_factory import ModelFactory
from config import Config as config


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

    # create live model store
    live_model_store = utils.get_live_models()

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
        return(json.dumps(ModelFactory.get_available_model_names()))

    # backtesting
    @app.route('/backtest', methods=['POST'])
    def backtest():
        try:
            post_data = request.get_json()
            post_metadata = post_data['metadata']
            app.logger.info('POST metada:\n%s', post_metadata)
            # If this is a correct model_name
            if(ModelFactory.model_is_existed(name=post_metadata['model_name'])):
                app.logger.info('Creating new model...')
                my_model = ModelFactory.create_model(
                    model_type=post_metadata['model_type'],
                    model_name=post_metadata['model_name'],
                    candle_size=post_metadata['candle_size'],
                    market_info=post_metadata['market_info'],
                    train_daterange=post_metadata['train_daterange'],
                    test_daterange=post_metadata['backtest_daterange'],
                    lag=post_metadata['lag'],
                    rolling_step=post_metadata['rolling_step'],
                    features=post_metadata['features'],
                    label=post_metadata['label'])

                raw_result = my_model.get_raw_data()
                x_train, y_train, x_rolling, y_rolling, x_predict = my_model.transform_data(
                    raw_result)
                y_predict = np.array([])

                # Backtest time!!
                app.logger.info('Predicting...')

                if (my_model.model_type == "fixed"):
                    my_model.train(x_train, y_train)
                    y_predict = my_model.predict(x_predict)
                elif (my_model.model_type == "rolling"):
                    # in case don't have enough data to roll
                    # stop re-trainning from that point
                    out_of_rolling = False

                    while (len(x_predict) != 0):
                        print('Number of predictions left: %10d' %
                              len(x_predict), end='\r')

                        if (not out_of_rolling):
                            my_model.train(x_train, y_train)

                        new_predictions = my_model.predict(
                            x_predict[:my_model.rolling_step])
                        actual_predictions_length = len(new_predictions)

                        y_predict = np.append(y_predict, new_predictions)

                        if(not out_of_rolling):
                            if(len(x_rolling) < actual_predictions_length):
                                out_of_rolling = True
                            else:
                                # perform sliding window:
                                #   append new rows from x_rolling, y_rolling to x_train, y_train
                                #   also remove old ones from x_train, y_train
                                x_train = np.append(
                                    x_train[actual_predictions_length:], x_rolling[:actual_predictions_length], axis=0)
                                y_train = np.append(
                                    y_train[actual_predictions_length:], y_rolling[:actual_predictions_length])
                                #   after that, removes old ones from x_rolling, y_rolling too
                                x_rolling = x_rolling[actual_predictions_length:]
                                y_rolling = y_rolling[actual_predictions_length:]
                        else:
                            app.logger.warning(
                                'Not enough data to perform rolling, model will stop retraining from now.')

                        # finally, shift x_predict
                        x_predict = x_predict[actual_predictions_length:]
                    print('Number of predictions left: %10d' % len(x_predict))
                else:
                    return 'Invalid model_type: %s' % my_model.model_type, 400

                # maximum profit, FOR TESTING PURPOSE ONLY
                y_predict = pd.DataFrame(raw_result['test']['data'])[[my_model.label]].values.reshape(
                    -1) if ('max_test' in post_metadata and post_metadata['max_test']) else y_predict

                # Send result
                result = {}
                for i in range(len(y_predict)):
                    result['{}'.format(raw_result['test']['data'][i]['start'])] = int(
                        y_predict[i])
                return json.dumps(result)
            # Return 404, model_name not found
            else:
                return 'Invalid model_name: %s' % (post_metadata['model_name']), 404

        except KeyError as e:
            app.logger.error(e)
            return 'Invalid JSON schema, please provide enough and correct params.', 400
        except Exception as e:
            app.logger.error(e)
            return str(e), 400

    # live trading
    @app.route('/live', methods=['POST'])
    def live_trading():
        try:
            # Get request JSON
            post_data = request.get_json()
            app.logger.info('POST data:\n%s', post_metadata)
            model_info = post_data['model_info']
            candle_start = post_data['candle_start']

            # Check model existence
            model_info_values = utils.get_string_values_inside(model_info)
            code_name = ModelFactory.calculate_code_name(model_info_values)

            if (code_name in live_model_store):
                # Use existing model in store
                app.logger.info('Using existing model: ' + code_name)
                model = live_model_store[code_name]
                # Update and re-train if necessary
                model.update(candle_start)
            else:
                # Create new model and add to store
                app.logger.info('Creating new model: ' + code_name)
                model = ModelFactory.create_model(
                    model_type=model_info['model_type'],
                    model_name=model_info['model_name'],
                    candle_size=model_info['candle_size'],
                    market_info=model_info['market_info'],
                    train_daterange=model_info['train_daterange'] if model_info['model_type'] == 'fixed' else None,
                    # test_daterange=model_info['backtest_daterange'],
                    lag=model_info['lag'],
                    rolling_step=model_info['rolling_step'],
                    features=model_info['features'],
                    label=model_info['label'])
                model.update(
                    candle_start, new_train_daterange=model.train_daterange)

            # Get pre_test, test from DB
            raw_pre_test, raw_test = model.get_raw_data2(
                candle_start, candle_start + model.candle_size*config.MINUTE_IN_MILLISECONDS)
            # Transform data and predict
            test_data = model.transform_data2(
                raw_pre_test, raw_test, cols_to_drop=['start', model.label])
            # Return result
            result = {}
            result['result'] = model.predict(test_data)[0]
            return json.dumps(result)
        except KeyError as e:
            app.logger.error(e)
            return 'Invalid JSON schema, please provide enough and correct params.', 400

    return app
