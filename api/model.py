from flask_restplus import Namespace, Resource, fields
from werkzeug.datastructures import FileStorage
from config import MODEL_META_DATA, DEFAULT_MODEL, MODELS
from core.backend import ModelWrapper, load_array

api = Namespace('model', description='Model information and inference operations')

model_meta = api.model('ModelMetadata', {
    'id': fields.String(required=True, description='Model identifier'),
    'name': fields.String(required=True, description='Model name'),
    'description': fields.String(required=True, description='Model description'),
    'license': fields.String(required=False, description='Model license')
})


@api.route('/metadata')
class Model(Resource):
    @api.doc('get_metadata')
    @api.marshal_with(model_meta)
    def get(self):
        """Return the metadata associated with the model"""
        return MODEL_META_DATA

predict_response = api.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'predictions': fields.List(fields.List(fields.Float), description='Predicted values for weather features')
})

# Set up parser for input data (http://flask-restplus.readthedocs.io/en/stable/parsing.html)
input_parser = api.parser()
input_parser.add_argument('file', type=FileStorage, location='files', required=True)
input_parser.add_argument('model',type=str, default=DEFAULT_MODEL, choices=MODELS)

@api.route('/predict')
class Predict(Resource):

    model_wrapper = ModelWrapper()

    @api.doc('predict')
    @api.expect(input_parser)
    @api.marshal_with(predict_response)
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
