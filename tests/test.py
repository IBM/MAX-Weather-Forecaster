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
    invalid_r = run_model(uv_model, model_endpoint + "?model=invalid")

    assert mv_r.status_code == 200
    assert ms_r.status_code == 200
    assert uv_r.status_code == 200
    assert invalid_r.status_code == 400

    mv_json = mv_r.json()
    ms_json = ms_r.json()
    uv_json = uv_r.json()

    assert mv_json['status'] == 'ok'
    assert ms_json['status'] == 'ok'
    assert uv_json['status'] == 'ok'

    for prediction in mv_json['predictions']:
        # some of the results are crazy. They should be updated in the model
        assert 10 >= prediction[0] >= 0  # HOURLYVISIBILITY - should be 0-10
        assert 106 > prediction[1] > -15  # HOURLYDRYBULBTEMPF - should be between extremes recorded in nyc
        assert 104 > prediction[2] > -13  # HOURLYWETBULBTEMPF - should be between extremes recorded in nyc
        assert 95 > prediction[3] > -30  # HOURLYDewPointTempF - should be between extremes recorded in nyc

        if prediction[4] < 100:  # when HOURLYRelativeHumidity is less than 100% (virtually always in real life)...
            assert round(prediction[1]) >= round(prediction[2])  # ... then HOURLYDRYBULBTEMPF > HOURLYWETBULBTEMPF

        assert 101 > prediction[4] >= 0  # HOURLYRelativeHumidity - apparently this CAN be > 100%
        assert 80 > prediction[5] >= 0  # HOURLYWindSpeed - should be non-negative
        assert 31 > prediction[6] > 28  # HOURLYStationPressure
        assert 31 > prediction[7] > 28  # HOURLYSeaLevelPressure
        assert 3 > prediction[8] >= 0  # HOURLYPrecip
        assert 31 > prediction[9] > 28  # HOURLYAltimeterSetting
        assert abs(prediction[9] - prediction[7]) < 0.1  # HOURLYAltimeterSetting ~= HOURLYSeaLevelPressure
        assert 1 > prediction[10] > -1 # HOURLYWindDirectionSin - should be -1 to 1
        assert 1 > prediction[11] > -1 # HOURLYWindDirectionCos - should be -1 to 1
        assert prediction[12] in [0,1] # HOURLYPressureTendencyIncr - should be 0 or 1
        assert prediction[13] in [0,1] # HOURLYPressureTendencyDecr - should be 0 or 1
        assert prediction[14] in [0,1] # HOURLYPressureTendencyCons - should be 0 or 1

    for prediction in ms_json['predictions']:
        # predicted HOURLYDRYBULBTEMPF for the next 48 hours
        assert 106 > max(prediction) # hottest recorded in nyc
        assert -15 < min(prediction) # coldest recorded in nyc
        assert len(prediction) == 48

    for prediction in uv_json['predictions']:
        # predicted HOURLYDRYBULBTEMPF for the next hour
        assert 106 > max(prediction) # hottest recorded in nyc
        assert -15 < min(prediction) # coldest recorded in nyc


if __name__ == '__main__':
    pytest.main([__file__])
