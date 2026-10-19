# Flight-Delay-Forecasting

The objective of this project is to develop a deep learning framework for flight delay forecasting, inspired by the multi-structural approach proposed in [*Aeolus: A Multi-structural Flight Delay Dataset* (Xu et al., 2025)](https://arxiv.org/pdf/2510.26616). The project investigates how heterogeneous information sources, including static flight attributes, weather conditions, and sequential flight-chain dynamics, can be exploited to predict delay propagation.

The proposed framework consists of three main components:

1. **Tabular modeling:** implementation of standalone machine learning and deep learning approaches based on static flight features. A LightGBM baseline and a Multi-Layer Perceptron (MLP) model are developed to evaluate the contribution of tabular information.

2. **Sequential modeling:** development of an LSTM-based model operating on flight-chain sequences to capture temporal delay propagation patterns between consecutive flights operated by the same aircraft, together with associated weather information.

3. **Multimodal fusion:** integration of the tabular and sequential representations through an LSTM+MLP fusion architecture, combining the complementary information learned from static flight characteristics and temporal flight dynamics.

The project includes the complete machine learning workflow, including data preprocessing, feature engineering, exploratory analysis, model development, training optimization, and evaluation of both classification and regression performance for flight delay prediction.

The final report, giving a detailed explanation of the theoretical background, approach followed, and results, is included as `Project_Report.pdf`. 

Detailed results and final model weights are available in folders `Results/` and `Models/`. 

## Repository structure

```
.
├── Material/                     # Reference paper and general report structure
├── Models_MLP/                   # Final trained model weights - MLP for fusion only
├── Notebooks/                    # Training and evaluation notebooks (and notebook for plotting)
├── PreProcessing/                # Script and notebooks for data cleaning and feature engineering
├── Results_Models_advanced/      # Final trained model weights for baseline LSTM and MLP-LSTM models; (also some results, plots, and notebook for plotting)
├── Project_Report.pdf            # Full project report
├── Requirements.txt              # Python dependencies
└── README.md
```

## Developers

* Kevin Brugnera (main contributions: MLP+LSTM fusion model)
* Sara Pasquato (main contributions: tabular LightGBM and MLP models, both baseline and for fusion)
* Libero Pollini (main contributions: dataset preprocessing, baseline LSTM model)


## Acknowledgements

The approach in data preprocessing, and partly in creating the baseline models, follows the one proposed by the creators of the [Aeolus dataset repo](https://github.com/Flnny/Delay-data).

This work was done as the final project for the course "Neural Networks and Deep Learning" at University of Padova held by professor Jacopo Pegoraro in the academic year 2025-26.

Note that raw and processed data are not included due to size; `PreProcessing/` documents the pipeline.

## Declaration of AI use

Generative artificial intelligence models were used throughout the project for improving productivity, including for drafting the final report and reordering this repository. All code was carefully reviewed and tested by the authors.

