# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'Model Asset Exchange Server'
API_DESC = 'An API for serving models'
API_VERSION = '0.1'

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
    'type': 'time-series-prediction',
    'license': '{}'.format(MODEL_LICENSE)
}
