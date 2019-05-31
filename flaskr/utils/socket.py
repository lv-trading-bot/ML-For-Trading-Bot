import socketio
import uuid
import json
import logging
from config import Config

sio = socketio.Client()
logger = logging.getLogger(Config.APP_LOGGER_NAME)


def get_id():
    try:
        # load previously created id (if any)
        with open(Config.ID_JSON_FILE) as json_file:
            data = json.load(json_file)
        if (data['id'] is None or data['id'] == ''):
            raise Exception()
        else:
            print('Using previous ID %s' % data['id'])
            return data['id']
    except Exception as e:
        print('Cannot not load previous ID, creating new one...')
        # create new id and save it
        data = {
            'id': 'ML_Server_' + uuid.uuid4().hex
        }
        with open(Config.ID_JSON_FILE, 'w') as outfile:
            json.dump(data, outfile)
        return data['id']


@sio.on('connect')
def on_connect():
    print('Socket: I\'m connected!')

    # data to be sent
    type = 'system'
    random_id = get_id()

    sio.emit('onConnect', (type, random_id))


@sio.on('disconnect')
def on_disconnect():
    logger.info('Socket: I\'m disconnected!')

