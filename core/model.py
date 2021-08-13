from maxfw.model import MAXModelWrapper

import tensorflow as tf
import logging
from config import DEFAULT_MODEL_PATH, MODELS, MODEL_META_DATA as model_meta
import os
from keras.models import load_model
import numpy as np
import joblib

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
        self.graph = tf.compat.v1.get_default_graph()
        self.model = load_model(model_path)

        # load scaler
        self._load_scaler(model, path)
        self._set_shape(model)

    def _set_shape(self, model):
        # Reshape data according to which model is being used
        if model == 'univariate':
            self.shape = (11452, 24, 12)
        elif model == 'multistep':
            self.shape = (11445, 48, 12)
        elif model == 'multivariate':
            self.shape = (11452, 24, 12)
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
            pred_zeroes = np.zeros((len(preds),12))
            pred_zeroes[:,1] = preds.reshape(-1)
            pred_zeroes = self.scaler.inverse_transform(pred_zeroes)
            rescaled_preds = pred_zeroes[:,1].reshape(len(pred_zeroes),1) # returns 2d array where first axis has 11452 entries and second axis has 1 timestep
        elif self.model_name == 'multistep':
            preds_list = []
            for i in range(48):
                pred_zeroes = np.zeros((len(preds),12)) 
                pred_zeroes[:,1] = preds[:,i].reshape(-1) 
                pred_zeroes = self.scaler.inverse_transform(pred_zeroes)
                pred_timestep = pred_zeroes[:,1]
                preds_list.append(pred_timestep)
            rescaled_preds = np.stack(preds_list, axis=1) # returns 2d array where first axis has 11445 entries and second axis has 48 timesteps
        elif self.model_name == 'multivariate':
            rescaled_preds = self.scaler.inverse_transform(preds) # returns 2d array where first axis has 11452 entries and second axis has 12 weather features
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

    def _predict(self, x, model):
        logger.info('Predicting from model: {}'.format(model))
        return self.models[model].predict(x)
