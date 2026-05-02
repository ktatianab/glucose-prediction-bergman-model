import os
import pickle #Cargar los normalizadores guardados, es decir scaler_X.pkl y scaler_y.pkl.
import numpy as np
import pandas as pd
import torch

from src.visualization.plots import ( plot_real_vs_predicted_time,  plot_real_vs_predicted_scatter)
from src.models.narx_network import NARXNetwork
from src.evaluation.metrics import (calculate_mae, calculate_mse, calculate_rmse, calculate_r2)

'''
    Este archivo se evalua la red neuronal NARX contra el modelo original de Bergman.
    Compara:
        y_true_test vs y_pred_nn

'''

'''
    Función principal para evaluar la red neuronal NARX contra el modelo original de Bergman.
'''
def evaluate_neural_network(
    test_data_path="src/data/processed/test_data.npz",
    model_path="saved_models/neural_network.pt",
    scaler_X_path="saved_models/scaler_X.pkl",
    scaler_y_path="saved_models/scaler_y.pkl",
    metrics_path="results/metrics/nn_metrics.csv"
):
    
    print("Cargando datos de prueba...")

    test_data = np.load(test_data_path)
    X_test = test_data["X"]
    y_true_test = test_data["y"]


    #Se cargan los normalizadores que fueron guardados durante el entrenamiento.
    print("Cargando scalers...")

    with open(scaler_X_path, "rb") as f:
        scaler_X = pickle.load(f)

    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)

   
    #Normalizar X_test

    X_test_scaled = scaler_X.transform(X_test)

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    
    # Crear modelo NARXNetwork
    
    input_dim = X_test.shape[1]       # normalmente 5
    output_dim = y_true_test.shape[1] # normalmente 3

    model = NARXNetwork(
        input_dim=input_dim,
        output_dim=output_dim
    )

    
    # Cargar pesos entrenados

    print("Cargando modelo entrenado...")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()


    # Predecir con la red neuronal

    print("Evaluando red neuronal...")

    #desactiva el cálculo de gradientes, porque no se está entrenando, solo evaluando. 
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    # La red predice datos normalizados, por eso se desnormaliza
    y_pred_nn = scaler_y.inverse_transform(y_pred_scaled)


    #  Calcular métricas

    mae = calculate_mae(y_true_test, y_pred_nn)
    mse = calculate_mse(y_true_test, y_pred_nn)
    rmse = calculate_rmse(y_true_test, y_pred_nn)
    r2 = calculate_r2(y_true_test, y_pred_nn)

    metrics_table = pd.DataFrame({
        "model": ["Neural Network"],
        "MAE": [mae],
        "MSE": [mse],
        "RMSE": [rmse],
        "R2": [r2]
    })


    # Guardar tabla de métricas

    os.makedirs("results/metrics", exist_ok=True)

    metrics_table.to_csv(metrics_path, index=False)

    print("Métricas guardadas en:", metrics_path)
    print(metrics_table)

    plot_real_vs_predicted_time(y_true_test, y_pred_nn)
    plot_real_vs_predicted_scatter(y_true_test, y_pred_nn)

    return metrics_table