import os
import json
import requests
from StateFarm.ML_tasks.solutions.utils import read_json


# Declare file path
basedir = os.path.abspath(os.path.dirname(__file__))
json_data = basedir + '/test/data.json'


data = read_json(json_data)

url = 'http://localhost:1313/predict'
r = requests.post(url, json=data)
print(r.json())



