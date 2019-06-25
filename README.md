[![Build Status](https://travis-ci.org/IBM/MAX-Weather-Forecaster.svg?branch=master)](https://travis-ci.org/IBM/MAX-Weather-Forecaster) [![Website Status](https://img.shields.io/website/http/max-weather-forecaster.max.us-south.containers.appdomain.cloud/swagger.json.svg?label=api+demo)](http://max-weather-forecaster.max.us-south.containers.appdomain.cloud/)

[<img src="docs/deploy-max-to-ibm-cloud-with-kubernetes-button.png" width="400px">](http://ibm.biz/max-to-ibm-cloud-tutorial)

# IBM Developer Model Asset Exchange: Weather Forecaster

This repository contains code to instantiate and deploy a weather forecasting model. The model takes hourly weather data
(as a Numpy array of various weather features, in text file format) as input and returns hourly weather predictions for
a specific target variable or variables (such as temperature or windspeed).

Three models have been included with this repository, all trained by the [CODAIT team](codait.org) on
[National Oceanic and Atmospheric Administration](https://www.ncdc.noaa.gov) local climatological data originally
collected by JFK airport. All three models use an LSTM recurrent neural network architecture. You can specify which
model you wish to use when making requests to the API (see [Use the Model](#3-use-the-model) below for more details).

A description of the weather variables used to train the models is set out below.

| Variable                 | Description           |
|---------------          | ----------------------|
| HOURLYVISIBILITY        | Distance from which an object can be seen. |
| HOURLYDRYBULBTEMPF      | Dry bulb temperature (degrees Fahrenheit). Most commonly reported standard temperature. |
| HOURLYWETBULBTEMPF      | Wet bulb temperature (degrees Fahrenheit).  |
| HOURLYDewPointTempF     | Dew point temperature (degrees Fahrenheit). |
| HOURLYRelativeHumidity  | Relative humidity (percent). |
| HOURLYWindSpeed         | Wind speed (miles per hour). |
| HOURLYWindDirection     | Wind direction from true north using compass directions. |
| HOURLYStationPressure   | Atmospheric pressure (inches of Mercury; or 'in Hg'). |
| HOURLYPressureTendency  | Pressure tendency, indicating pressure change over most recent 3 hour period. |
| HOURLYSeaLevelPressure  | Sea level pressure (in Hg). |
| HOURLYPrecip            | Total precipitation in the past hour (in inches). |
| HOURLYAltimeterSetting  | Atmospheric pressure reduced to sea level using temperature profile of the “standard” atmosphere (in Hg). |

For further details on the weather variables see the [US Local Climatological Data Documentation](https://www1.ncdc.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf)

Each model returns a different format for its predictions:
* *Univariate Model*: returns a prediction of dry bulb temperature (`HOURLYDRYBULBTEMPF`), for the next hourly time step, for each input data point
* *Multivariate Model*: returns predictions for all 12 weather variables, for the next hourly time step, for each input data point
* *Multistep Model*: returns predictions of dry bulb temperature (`HOURLYDRYBULBTEMPF`), for the next 48 hourly time steps, for each input data point

The model files are provided as part of this repository in the [`assets/models`](assets/models) folder. The code in this
repository deploys the model as a web service in a Docker container. This repository was developed as part of the
[IBM Code Model Asset Exchange](https://developer.ibm.com/code/exchanges/models/) and the public API is powered by
[IBM Cloud](https://ibm.biz/Bdz2XM).
## Model Metadata

| Domain        | Application           | Industry       | Framework  | Training Data           | Input Data Format |
|---------------|-----------------------|----------------|------------|-------------------------|-------------------|
| Weather       | Time Series Prediction | General | TensorFlow / Keras | [JFK Airport Weather Data, NOAA](https://www.ncdc.noaa.gov/cdo-web/datasets/LCD/stations/WBAN:94789/detail) | CSV |

* Data from [US Local Climatological Data](https://www.ncdc.noaa.gov/cdo-web/datatools/lcd), National Climatic Data Center, National Oceanic & Atmospheric Administration

## References

Literature and Documentation
* [LSTMs in Keras](https://keras.io/layers/recurrent/#lstm)
* [Time Series Prediction with RNNs](https://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html)
* _S. Hochreiter, J. Schmidhuber_ ["Long Short Term Memory"](http://www.bioinf.jku.at/publications/older/2604.pdf), Neural Computation 1997

Related Repositories
* [TensorFlow Tutorials for Time Series](https://github.com/tgjeon/TensorFlow-Tutorials-for-Time-Series)
* [Tensorflow LSTM Regression](https://github.com/mouradmourafiq/tensorflow-lstm-regression)

## Licenses

| Component | License | Link  |
| ------------- | --------  | -------- |
| This repository | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [LICENSE](LICENSE) |
| Model Weights | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [LICENSE](LICENSE) |
| Test Assets | No restriction | [Asset README](assets/README.md) |

## Prerequisites

* `docker`: The [Docker](https://www.docker.com/) command-line interface. Follow the [installation instructions](https://docs.docker.com/install/) for your system.
* The minimum recommended resources for this model is 2GB Memory and 2 CPUs.

# Steps

1. [Deploy from Docker Hub](#deploy-from-docker-hub)
2. [Deploy on Kubernetes](#deploy-on-kubernetes)
3. [Run Locally](#run-locally)

## Deploy from Docker Hub

To run the docker image, which automatically starts the model serving API, run:

```
$ docker run -it -p 5000:5000 codait/max-weather-forecaster
```

This will pull a pre-built image from Docker Hub (or use an existing image if already cached locally) and run it.
If you'd rather checkout and build the model locally you can follow the [run locally](#run-locally) steps below.

## Deploy on Kubernetes

You can also deploy the model on Kubernetes using the latest docker image on Docker Hub.

On your Kubernetes cluster, run the following commands:

```
$ kubectl apply -f https://github.com/IBM/MAX-Weather-Forecaster/raw/master/max-weather-forecaster.yaml
```

The model will be available internally at port `5000`, but can also be accessed externally through the `NodePort`.

## Run Locally

1. [Build the Model](#1-build-the-model)
2. [Deploy the Model](#2-deploy-the-model)
3. [Use the Model](#3-use-the-model)
4. [Development](#4-development)
5. [Clean Up](#5-cleanup)

### 1. Build the Model

Clone this repository locally. In a terminal, run the following command:

```
$ git clone https://github.com/IBM/MAX-Weather-Forecaster.git
```

Change directory into the repository base folder: 

```
$ cd MAX-Weather-Forecaster
```

To build the docker image locally, run:

```
$ docker build -t max-weather-forecaster .
```

_Note_ that currently this docker image is CPU only (we will add support for GPU images later).

## 2. Deploy the Model

To run the docker image, which automatically starts the model serving API, run:

```
$ docker run -it -p 5000:5000 max-weather-forecaster
```

## 3. Use the Model

The API server automatically generates an interactive Swagger documentation page. Go to `http://localhost:5000` to load it. From there you can explore the API and also create test requests.

Use the `model/predict` endpoint to load a test data file and get predictions for the relevant weather target variable (or variables) from the API. You can use one of the test files from the `assets/lstm_weather_test_data` folder, after unzipping the test data archive by running the following command:

```
$ tar -zxvf assets/lstm_weather_test_data.tar.gz -C assets
```

![Swagger Screenshot](/docs/swagger-screenshot.png "Swagger Screenshot")

You can also test it on the command line, for example to test the univariate model:
```
$ curl -F "file=@assets/lstm_weather_test_data/univariate_model_test_data.txt" -XPOST http://localhost:5000/model/predict
```

You can select one of the three available models used to make predictions by setting the `model` request parameter to one of: `univariate` (default), `multivariate`, or `multistep`. _Note_ that each model takes in different weather datasets. After loading a particular model, you must predict only on the accompanying test dataset (e.g. `univariate` must predict on `univariate_model_test_data.txt`).

For example, to test the multivariate model:
```
$ curl -F "file=@assets/lstm_weather_test_data/multivariate_model_test_data.txt" -XPOST http://localhost:5000/model/predict?model=multivariate
```

To test the multi-step model:
```
$ curl -F "file=@assets/lstm_weather_test_data/multistep_model_test_data.txt" -XPOST http://localhost:5000/model/predict?model=multistep
```

You should see a JSON response like that below for the `multistep` test data, where `predictions` contains the predicted dry bulb temperature (in F) for each of the next 48 hours, for each input data point.

```
{
  "status": "ok",
  "predictions": [
    [
      77.51201432943344,
      76.51381462812424,
      75.0168582201004,
      73.84445126354694,
      72.79087746143341,
      71.71804094314575,
      70.97693882882595,
      70.44060184061527,
      69.89843893051147,
      69.35454525053501,
      69.04163710772991,
      68.70432360470295,
      68.37075608968735,
      68.20421539247036,
      68.01852786540985,
      67.6653740555048,
      67.27566187083721,
      67.0398361980915,
      66.69407051801682,
      66.9289058893919,
      67.19844545423985,
      67.65162572264671,
      68.30480472743511,
      69.37090930342674,
      70.37226051092148,
      71.57235226035118,
      72.68855434656143,
      73.91224025189877,
      74.65138283371925,
      75.09161844849586,
      75.30447003245354,
      75.04770956933498,
      74.93723678588867,
      74.27759975194931,
      73.82458955049515,
      73.32358133792877,
      72.66812674701214,
      71.75925283133984,
      71.28871068358421,
      70.66486597061157,
      70.06835387647152,
      69.74887031316757,
      69.49707941710949,
      69.26406812667847,
      68.87126012146473,
      68.60496838390827,
      68.39429907500744,
      68.03596951067448
    ],
    ...
}
```

## 4. Development

To run the Flask API app in debug mode, edit `config.py` to set `DEBUG = True` under the application settings. You will
then need to rebuild the Docker image (see [step 1](#1-build-the-model)).

## 5. Cleanup

To stop the Docker container, type `CTRL` + `C` in your terminal.
