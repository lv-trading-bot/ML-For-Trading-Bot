import os
import joblib
import json
import numpy as np
import math
import time
import copy
import pandas as pd

from flask import Flask, request, abort, g
from flaskr.utils import Utils as utils
from flaskr.models.model_factory import ModelFactory
from flaskr.utils.socket import sio as sio_client
from config import Config as config

import logging
logging.basicConfig(
    format='%(asctime)s (%(levelname)s): %(message)s ', datefmt='%m/%d/%Y %I:%M:%S%p')


def create_app(test_config=None):
    # connect to socket server
    sio_client.connect(config.SOCKET_URL)

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
    live_model_store = ModelFactory.get_live_models()

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
            train_daterange = post_metadata['train_daterange']
            test_daterange = post_metadata['backtest_daterange']

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
                    lag=post_metadata['lag'],
                    rolling_step=post_metadata['rolling_step'],
                    features=post_metadata['features'],
                    label=post_metadata['label'])

                # # old
                # raw_pre_train, raw_train = my_model.get_raw_data(
                #     train_daterange['from'], train_daterange['to'])
                # test_size = test_daterange['to'] - test_daterange['from']
                # raw_rolling = my_model.get_candles_by_daterange(
                #     train_daterange['to'], train_daterange['to'] + test_size)

                # new: get h more candles
                test_size = test_daterange['to'] - test_daterange['from']
                raw_pre_train, raw_train, raw_rolling = my_model.get_raw_train_data_for_backtest(
                    train_daterange, rolling_size=test_size)

                raw_pre_test, raw_test = my_model.get_raw_data(
                    test_daterange['from'], test_daterange['to'])

                # raw_test copy for later use
                raw_test_copy = copy.deepcopy(raw_test)

                # Backtest time!!
                app.logger.info('Predicting...')
                y_predict = np.array([])

                if (my_model.model_type == "fixed"):
                    x_train, y_train = my_model.prepare_data(
                        raw_pre_train, raw_train, for_training=True)
                    x_test, y_test = my_model.prepare_data(
                        raw_pre_test, raw_test)

                    my_model.train(x_train, y_train)
                    y_predict = my_model.predict_proba(x_test)

                elif (my_model.model_type == "rolling"):
                    rolling_step = my_model.rolling_step

                    while(len(raw_test) != 0):
                        print('Number of predictions left: %10d' %
                              len(raw_test), end='\r')

                        # prepare data and train
                        x_train, y_train = my_model.prepare_data(
                            raw_pre_train, raw_train, for_training=True)
                        x_test, y_test = my_model.prepare_data(
                            raw_pre_test, raw_test[:rolling_step])
                        my_model.train(x_train, y_train)
                        y_predict = np.append(
                            y_predict, my_model.predict_proba(x_test))

                        # perform sliding window for train data:
                        # merge all for easy manipulation, remove old candles
                        dropped = (raw_pre_train + raw_train +
                                   raw_rolling)[rolling_step:]
                        # split back
                        raw_pre_train = dropped[:len(raw_pre_train)]
                        raw_train = dropped[len(raw_pre_train): len(
                            raw_pre_train)+len(raw_train)]
                        raw_rolling = dropped[len(
                            raw_pre_train) + len(raw_train):]

                        # perform sliding window for test data: the same
                        dropped2 = (raw_pre_test + raw_test)[rolling_step:]
                        raw_pre_test = dropped2[:len(raw_pre_test)]
                        raw_test = dropped2[len(raw_pre_test):]

                    print('Number of predictions left: %10d' % len(raw_test))

                else:
                    return 'Invalid model_type: %s' % my_model.model_type, 400

                # maximum profit, FOR TESTING PURPOSE ONLY
                y_predict = pd.DataFrame(raw_test_copy)[[my_model.label]].values.reshape(
                    -1).tolist() if ('max_test' in post_metadata and post_metadata['max_test']) else y_predict

                # Send result
                result = {}
                for i in range(len(y_predict)):
                    result['{}'.format(
                        raw_test_copy[i]['start'])] = y_predict[i]
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
    def live():
        try:
            # Get request JSON
            post_data = request.get_json()
            app.logger.info('POST data:\n%s', post_data)
            model_info = post_data['model_info']
            candle_start = post_data['candle_start']

            # Create first -> to calculate code_name
            model = ModelFactory.create_model(
                model_type=model_info['model_type'],
                model_name=model_info['model_name'],
                candle_size=model_info['candle_size'],
                market_info=model_info['market_info'],
                train_daterange=model_info['train_daterange'] if 'train_daterange' in model_info else None,
                lag=model_info['lag'] if 'lag' in model_info else None,
                rolling_step=model_info['rolling_step'] if 'rolling_step' in model_info else None,
                features=model_info['features'],
                label=model_info['label'])

            # If model_code_name is specified, use it
            if ('model_code_name' in post_data):
                code_name = post_data['model_code_name']
            else:
                code_name = model.code_name

            # Check model existence in store
            if (code_name in live_model_store):
                app.logger.info('Using existing model: ' + code_name)
                model = live_model_store[code_name]
            else:
                app.logger.info('Creating new model: ' + code_name)
                # Add new model to store
                live_model_store[code_name] = model
                model.train_by_daterange()

            # Update and re-train if necessary
            model.update_by_candle_start(candle_start)
            # After all, save model!
            model.save(config.LIVE_MODELS_DIR)

            # Prepare test data
            raw_pre_test, raw_test = model.get_raw_data(
                candle_start, candle_start + model.candle_size * config.MINUTE_IN_MILLISECONDS)
            x_test, y_test = model.prepare_data(raw_pre_test, raw_test)

            # Predict
            y_predict = model.predict_proba(x_test)

            result = {}
            result['result'] = y_predict[0]

            return json.dumps(result)

        except KeyError as e:
            app.logger.error(e)
            return 'Invalid JSON schema, please provide enough and correct params.', 400
        except Exception as e:
            app.logger.error(e)
            return str(e), 400

    return app
