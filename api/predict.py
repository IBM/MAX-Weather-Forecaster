from core.model import ModelWrapper, load_array

from maxfw.core import MAX_API, PredictAPI

from flask_restplus import Namespace, Resource, fields
from werkzeug.datastructures import FileStorage
from config import MODEL_META_DATA, DEFAULT_MODEL, MODELS


predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'predictions': fields.List(fields.List(fields.Float), description='Predicted values for weather features')
})

# Set up parser for input data (http://flask-restplus.readthedocs.io/en/stable/parsing.html)
input_parser = MAX_API.parser()
input_parser.add_argument('file', type=FileStorage, location='files', required=True,
    help='Input data to use for prediction, in the form of a numpy array txt file')
input_parser.add_argument('model',type=str, default=DEFAULT_MODEL, choices=MODELS,
    help='Underlying model to use for prediction')


class ModelPredictAPI(PredictAPI):

    model_wrapper = ModelWrapper()

    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        """Make a prediction given input data"""
        result = {'status': 'error'}

        args = input_parser.parse_args()
        input_data = args['file']
        model = args['model']

        input_array = load_array(input_data)
        preds = self.model_wrapper.predict(input_array, model)

        result['predictions'] = preds
        result['status'] = 'ok'

        return result
