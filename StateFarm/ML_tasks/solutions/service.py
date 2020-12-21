#!flask/bin/src
"""
Flask application to serve Machine Learning models
"""

import os
import pandas as pd
from flask import Flask, jsonify, request
from logging.handlers import RotatingFileHandler
import logging
import pickle
from time import time
from StateFarm.ML_tasks.solutions.utils import read_json
from flask_expects_json import expects_json


# Read env variables
DEBUG = os.environ.get('DEBUG', True)
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'local')
SERVICE_START_TIMESTAMP = time()
# Create Flask Application
app = Flask(__name__)

# Declare file path
basedir = os.path.abspath(os.path.dirname(__file__))
models_dir = os.path.join(basedir, 'models')
json_schema_dir = basedir + '/test/schema.json'
schema = read_json(json_schema_dir)
# Load Logistic Regression model
model = pickle.load(open(models_dir + '/logit.pkl', 'rb'))
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
@expects_json(schema)
def predict():
    # try:
    #     test_json = request.get_json()
    #     test = pd.read_json(test_json, orient='records')
    #
    #     # To resolve the issue of TypeError: Cannot compare types 'ndarray(dtype=int64)' and 'str'
    #     test['Dependents'] = [str(x) for x in list(test['Dependents'])]
    #
    #     # Getting the Loan_IDs separated out
    #     loan_ids = test['Loan_ID']
    #
    # except Exception as e:
    #     raise e
    # if request.method == 'POST':
    if request.method=='POST':
        data = request.get_json()

        return jsonify(str("Successfully stored  " + str(data)))

    return jsonify(test)


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
