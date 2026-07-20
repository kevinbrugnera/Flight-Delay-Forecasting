# Flight-Delay-Forecasting

The objective of the project is to develop a deep learning framework for flight delay forecasting, inspired by the multi-structural approach proposed in *Aeolus: A Multi-structural Flight Delay Dataset*. The aim is to investigate how different sources of information, including sequential flight patterns, weather conditions, and tabular flight features, can be combined to predict delay propagation.

The project workflow consists of three main steps:
1. Implementation of an LSTM-based model on the sequential component of the dataset ("flight chain") to capture delay propagation patterns along the flights operated by the same aircraft, together with weather information.
2. Integration of tabular features from the dataset to enrich the predictive model.
3. Extension of the framework with a CNN-based component to capture spatial relationships between airports.

The project includes data preprocessing, exploratory analysis, deep learning model development, and evaluation of forecasting performance.

### Developers

* Kevin Brugnera
* Sara Pasquato
* Libero Pollini
