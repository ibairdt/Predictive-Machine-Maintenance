# Predictive Machine Maintenance Project

The aim of this project is to create a machine learning model in order to predict whether a machine is likely to suffer a failure or not. The data used in this project was extracted for academic purposes from the following [Kaggle post](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification/data)

## Repository structure
The project is composed by the following folders, which include the necessary steps for programming an effective ML model:

1. `data` \
    `data / raw` includes the original dataset (predictive_maintenance.csv) as well as the slightly processed datasets with the binary and the categorical targets \
    `data / clean` includes the fully processed dataset used to programme the ML models, using only the dicotomic target \
    `data / incoming` includes a file composed by a subset of the original dataset, which is used as an example when making our first prediction with the Python script

2. `notebooks` folder composed by the following two Jupyter Notebooks \
    `notebooks / EDA.ipynb` this notebook contains the Exploratory Data Analysis, including the whole data preprocessing, its univariative and multivariative analyses, data visualisation, variable standardisation, sample resizing, etcetera. \
    `notebooks / model_programming.ipynb` this is the notebook where we programmed several ML models and measured their performances. Finally we binarise our resulting model, which will be used for future predictions.

3. `models` & `scalers` folders are composed by the Pickle files which include the binarisation of our resulting model, as well as the exported Standard Scaler, necessary for carrying out future predictions with the mentioned model.

4. `predictions` this is the folder where our model stores its predictions.

5. `production` folder including the prediction Python script, ready to be executed.