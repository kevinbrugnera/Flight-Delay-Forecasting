# Flight-Delay-Forecasting

The objective of this project is to develop a deep learning framework for flight delay forecasting, inspired by the multi-structural approach proposed in *Aeolus: A Multi-structural Flight Delay Dataset*. The project investigates how heterogeneous information sources, including static flight attributes, weather conditions, and sequential flight-chain dynamics, can be exploited to predict delay propagation.

The proposed framework consists of three main components:

1. **Tabular modeling:** implementation of standalone machine learning and deep learning approaches based on static flight features. A LightGBM baseline and a Multi-Layer Perceptron (MLP) model are developed to evaluate the contribution of tabular information.

2. **Sequential modeling:** development of an LSTM-based model operating on flight-chain sequences to capture temporal delay propagation patterns between consecutive flights operated by the same aircraft, together with associated weather information.

3. **Multimodal fusion:** integration of the tabular and sequential representations through an LSTM+MLP fusion architecture, combining the complementary information learned from static flight characteristics and temporal flight dynamics.

The project includes the complete machine learning workflow, including data preprocessing, feature engineering, exploratory analysis, model development, training optimization, and evaluation of both classification and regression performance for flight delay prediction.

### Developers

* Kevin Brugnera
* Sara Pasquato
* Libero Pollini