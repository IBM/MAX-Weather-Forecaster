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

from core.model import ModelWrapper
from maxfw.core import MAX_API, PredictAPI
from flask_restplus import fields
from werkzeug.datastructures import FileStorage
from config import DEFAULT_MODEL, MODELS


predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'predictions': fields.List(fields.List(fields.Float), description='Predicted values for weather features')
})

# Set up parser for input data (http://flask-restplus.readthedocs.io/en/stable/parsing.html)
input_parser = MAX_API.parser()
input_parser.add_argument('file', type=FileStorage, location='files', required=True,
                          help='Input data to use for prediction, in the form of a numpy array txt file')
input_parser.add_argument('model', type=str, default=DEFAULT_MODEL, choices=MODELS,
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

        preds = self.model_wrapper.predict(args)

        result['predictions'] = preds
        result['status'] = 'ok'

        return result
