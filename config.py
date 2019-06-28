#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'MAX Weather Forecaster'
API_DESC = 'An API for serving models'
API_VERSION = '1.1.0'

# Default model
MODEL_NAME = 'lstm_weather_forecaster'
DEFAULT_MODEL_PATH = 'assets/models'
MODEL_LICENSE = 'Apache 2'
MODELS = ['univariate', 'multistep', 'multivariate']
DEFAULT_MODEL = MODELS[0]

MODEL_META_DATA = {
    'id': '{}'.format(MODEL_NAME.lower()),
    'name': 'LSTM Weather Forecaster',
    'description': 'LSTM Weather Forecaster Model trained using TensorFlow and Keras on JFK weather time-series data',
    'type': 'Time Series Prediction',
    'license': '{}'.format(MODEL_LICENSE),
    'source': 'https://developer.ibm.com/exchanges/models/all/max-weather-forecaster/'
}
