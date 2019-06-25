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

import pytest
import requests


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'MAX Weather Forecaster'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'lstm_weather_forecaster'
    assert metadata['name'] == 'LSTM Weather Forecaster'
    assert metadata['description'] == 'LSTM Weather Forecaster Model trained using TensorFlow and Keras on JFK ' \
                                      'weather time-series data'
    assert metadata['license'] == 'Apache 2'
    assert metadata['type'] == 'Time Series Prediction'
    assert 'max-weather-forecaster' in metadata['source']


def run_model(file_path, url):
    with open(file_path, 'rb') as file:
        file_form = {'file': (file_path, file, 'text/plain')}
        r = requests.post(url=url, files=file_form)
        return r


def test_predict():
    model_endpoint = 'http://localhost:5000/model/predict'

    mv_model = 'assets/lstm_weather_test_data/multivariate_model_test_data.txt'
    ms_model = 'assets/lstm_weather_test_data/multistep_model_test_data.txt'
    uv_model = 'assets/lstm_weather_test_data/univariate_model_test_data.txt'

    mv_r = run_model(mv_model, model_endpoint + "?model=multivariate")
    ms_r = run_model(ms_model, model_endpoint + "?model=multistep")
    uv_r = run_model(uv_model, model_endpoint + "?model=univariate")

    assert mv_r.status_code == 200
    assert ms_r.status_code == 200
    assert uv_r.status_code == 200

    mv_json = mv_r.json()
    ms_json = ms_r.json()
    uv_json = uv_r.json()

    assert mv_json['status'] == 'ok'
    assert ms_json['status'] == 'ok'
    assert uv_json['status'] == 'ok'

    for prediction in mv_json['predictions']:
        # some of the results are crazy. They should be updated in the model
        assert 11 > prediction[0] > -1  # HOURLYVISIBILITY TODO - should be 0-10
        assert 100 > prediction[1] > 0  # HOURLYDRYBULBTEMPF
        assert 100 > prediction[2] > 0  # HOURLYWETBULBTEMPF
        assert (prediction[1] + 0.7) > prediction[2]  # HOURLYDRYBULBTEMPF > HOURLYWETBULBTEMPF TODO
        assert 101 > prediction[4] > 0  # HOURLYRelativeHumidity, apparently this CAN be > 100%
        assert 80 > prediction[5] > -3  # HOURLYWindSpeed
        assert 370 > prediction[6] > -6  # HOURLYWindDirection TODO - should be 0-360
        assert 31 > prediction[7] > 28  # HOURLYStationPressure
        assert 9 > prediction[8] > -1  # HOURLYPressureTendency TODO
        assert 31 > prediction[9] > 28  # HOURLYSeaLevelPressure
        assert 2 > prediction[10] > -0.5  # HOURLYPrecip TODO
        assert 31 > prediction[11] > 28  # HOURLYAltimeterSetting
        assert abs(prediction[11] - prediction[9]) < 0.1  # HOURLYAltimeterSetting ~= HOURLYSeaLevelPressure

    for prediction in ms_json['predictions']:
        # predicted HOURLYDRYBULBTEMPF for the next 48 hours
        assert 100 > max(prediction)
        assert 0 < min(prediction)
        assert len(prediction) == 48

    for prediction in uv_json['predictions']:
        # predicted HOURLYDRYBULBTEMPF for the next hour
        assert 100 > max(prediction)
        assert 0 < min(prediction)


if __name__ == '__main__':
    pytest.main([__file__])
