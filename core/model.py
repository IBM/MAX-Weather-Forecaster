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

from maxfw.model import MAXModelWrapper
import tensorflow as tf
import logging
from config import DEFAULT_MODEL_PATH, MODELS, MODEL_META_DATA as model_meta
from keras.models import load_model
import numpy as np
from sklearn.externals import joblib

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_array(input_data):
    return np.loadtxt(input_data)


class SingleModelWrapper(object):

    def __init__(self, model, path):
        self.model_name = model

        # load model
        model_path = '{}/{}_model'.format(path, model)
        logger.info(model_path)
        self.graph = tf.get_default_graph()
        self.model = load_model(model_path)

        # load scaler
        self._load_scaler(model, path)
        self._set_shape(model)

    def _set_shape(self, model):
        # Reshape data according to which model is being used
        if model == 'univariate':
            self.shape = (7509, 24, 15)
        elif model == 'multistep':
            self.shape = (7502, 48, 15)
        elif model == 'multivariate':
            self.shape = (7509, 24, 15)
        else:
            raise ValueError('Invalid model: {}'.format(self.model))

    def _load_scaler(self, model, path):
        # Load model appropriate scaler
        scaler_path = '{}/{}_model_scaler.save'.format(path, model)
        self.scaler = joblib.load(scaler_path)

    def _reshape_data(self, x):
        return x.reshape(self.shape)

    def _rescale_preds(self, preds):
        # Apply inverse transform for relevant model
        if self.model_name == 'univariate':
            pred_zeroes = np.zeros((len(preds), 15))
            pred_zeroes[:, 1] = preds.reshape(-1)
            pred_zeroes = self.scaler.inverse_transform(pred_zeroes)

            # returns 2d array where first axis has 7509 entries and second axis has 1 timestep
            rescaled_preds = pred_zeroes[:, 1].reshape(len(pred_zeroes), 1)
        elif self.model_name == 'multistep':
            preds_list = []
            for i in range(48):
                pred_zeroes = np.zeros((len(preds), 15))
                pred_zeroes[:, 1] = preds[:, i].reshape(-1)
                pred_zeroes = self.scaler.inverse_transform(pred_zeroes)
                pred_timestep = pred_zeroes[:, 1]
                preds_list.append(pred_timestep)

                # returns 2d array where first axis has 7502 entries and second axis has 48 timesteps
            rescaled_preds = np.stack(preds_list, axis=1)
        elif self.model_name == 'multivariate':
            # returns 2d array where first axis has 7509 entries and second axis has 12 weather features
            rescaled_preds = self.scaler.inverse_transform(preds)
        else:
            raise ValueError('Invalid model: {}'.format(self.model))

        return rescaled_preds

    def predict(self, x):
        reshaped_x = self._reshape_data(x)
        with self.graph.as_default():
            preds = self.model.predict(reshaped_x)
        rescaled_preds = self._rescale_preds(preds)
        return rescaled_preds


class ModelWrapper(MAXModelWrapper):

    MODEL_META_DATA = model_meta

    """Model wrapper for Keras models in SavedModel format"""
    def __init__(self, path=DEFAULT_MODEL_PATH):

        logger.info('Loading models from: {}...'.format(path))
        self.models = {}
        for model in MODELS:
            logger.info('Loading model: {}'.format(model))
            self.models[model] = SingleModelWrapper(model=model, path=path)

        logger.info('Loaded all models')

    def _predict(self, args):
        model = args['model']
        input_data = args['file']
        x = load_array(input_data)
        logger.info('Predicting from model: {}'.format(model))
        return self.models[model].predict(x)

    def _post_process(self, predictions):
        if predictions.shape[1] == 15:  # multivariate model
            predictions = np.array(predictions)
            predictions[:, 0] = np.clip(np.rint(predictions[:, 0]), 0, 10)  # HOURLYVISIBILITY - should be int 0-10
            predictions[:, 1] = np.clip(predictions[:, 1], -15, 106)  # HOURLYDRYBULBTEMPF - between extremes recorded in nyc
            predictions[:, 2] = np.clip(predictions[:, 2], -13, 104)  # HOURLYWETBULBTEMPF - between extremes recorded in nyc
            predictions[:, 3] = np.clip(predictions[:, 3], -30, 95)  # HOURLYDewPointTempF - between extremes recorded in nyc
            predictions[:, 4] = np.clip(predictions[:, 4], 0, 101)  # HOURLYRelativeHumidity - between extremes recorded in nyc
            predictions[:, 5] = np.maximum(predictions[:, 5], 0)  # HOURLYWindSpeed - should be non-negative
            predictions[:, 6] = np.clip(predictions[:, 6], 28, 31)  # HOURLYStationPressure
            predictions[:, 7] = np.clip(predictions[:, 7], 28, 31)  # HOURLYSeaLevelPressure
            predictions[:, 8] = np.maximum(predictions[:, 8], 0)  # HOURLYPrecip - should be non-negative
            predictions[:, 9] = np.clip(predictions[:, 9], 28, 31)  # HOURLYAltimeterSetting
            predictions[:, 10] = np.clip(predictions[:, 10], -1, 1)  # HOURLYWindDirectionSin - should be -1 to 1
            predictions[:, 11] = np.clip(predictions[:, 11], -1, 1)  # HOURLYWindDirectionCos - should be -1 to 1
            predictions[:, 12] = np.clip(np.rint(predictions[:, 12]), 0, 1)  # HOURLYPressureTendencyIncr - should be 0 or 1
            predictions[:, 13] = np.clip(np.rint(predictions[:, 13]), 0, 1)  # HOURLYPressureTendencyDecr - should be 0 or 1
            predictions[:, 14] = np.clip(np.rint(predictions[:, 14]), 0, 1)  # HOURLYPressureTendencyCons - should be 0 or 1

        return predictions
