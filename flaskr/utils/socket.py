import socketio
import json
import logging
from config import Config

sio = socketio.Client()
logger = logging.getLogger(Config.APP_LOGGER_NAME)


def get_id():
    return Config.ID


@sio.on('connect')
def on_connect():
    print('Socket: I\'m connected!')

    # data to be sent
    type = 'system'
    my_id = get_id()

    sio.emit('onConnect', (type, my_id))


@sio.on('disconnect')
def on_disconnect():
    logger.info('Socket: I\'m disconnected!')
