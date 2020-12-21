#!flask/bin/src
"""
Flask application to serve Machine Learning models
"""

import os
from flask import Flask, jsonify, request
from logging.handlers import RotatingFileHandler
import logging
from time import time

# Read env variables
DEBUG = os.environ.get('DEBUG', True)
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'local')
SERVICE_START_TIMESTAMP = time()
# Create Flask Application
app = Flask(__name__)
# Customize Flask Application
app.logger.setLevel(logging.DEBUG if DEBUG else logging.ERROR)
handler = RotatingFileHandler('foo.log', maxBytes=10000, backupCount=1)
app.logger.addHandler(handler)
app.logger.info(f'ENVIRONMENT: {ENVIRONMENT}')
app.logger.info('Loading model...')


# import json
# # from prediction import predict
# HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}
#
# # Create Flask Application
# app = Flask(__name__)
# @app.before_request
# def log_request_info():
#     app.logger.debug('Headers: %s', request.headers)
#     app.logger.debug('Body: %s', request.get_data())


@app.route('/', methods=['GET'])
def server_is_up():
    return jsonify({"status": 200,
                    'description': 'service is up'})


@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({"predict cost": 0})


#     @app.route('/predict', methods=['POST'])
#     def start():
#         # to_predict = request.json
#         #
#         # print(to_predict)
#         # pred = predict(to_predict)
#         return jsonify({"predict cost": 0 })
#     return app
#
# if __name__ == '__main__':
#     app = flask_app()
#     app.run(debug=True, host='0.0.0.0')

if __name__ == '__main__':
    app.run(
        debug=DEBUG,
        host=os.environ.get('HOST', 'localhost'),
        port=os.environ.get('PORT', '1313'))
