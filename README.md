# Artificial Pancreas: Bergman Model Identification using Neural Networks and ANFIS

## Overview
This project simulates the Bergman minimal model and trains machine learning models to approximate glucose-insulin dynamics.

## Technologies Used

- **Python**: main programming language used for simulation, model training, evaluation, and visualization.
- **NumPy**: numerical computing and array manipulation.
- **SciPy**: support for numerical methods and scientific computing.
- **PyTorch**: implementation and training of the neural network NARX model.
- **scikit-learn**: data preprocessing, train/test split, and evaluation metrics.
- **Matplotlib**: visualization of simulations, predictions, and model comparison results.
- **Pandas**: organization and export of metrics and results.
- **Joblib**: saving preprocessing objects such as scalers.
- **Git & GitHub**: version control and project portfolio hosting.

## Models
- Bergman Minimal Model
- Neural Network NARX
- ANFIS model

## Inputs and Outputs
Input: [G(k), X(k), I(k), u(k), D(k)]  
Output: [G(k+1), X(k+1), I(k+1)]

## Results
- MAE
- RMSE
- R²
- Prediction plots
- Real vs predicted scatter plots

## How to run
