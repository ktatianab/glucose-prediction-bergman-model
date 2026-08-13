import os
import pickle #Cargar los normalizadores guardados, es decir scaler_X.pkl y scaler_y.pkl.
import numpy as np
import pandas as pd
import torch

from src.visualization.plots import ( plot_real_vs_predicted_time,  plot_real_vs_predicted_scatter)
from src.models.narx_network import NARXNetwork
from src.models.anfis_model import ANFIS
from src.evaluation.metrics import (calculate_mae, calculate_mse, calculate_rmse, calculate_r2,
                                    calculate_decoupled_metrics, get_metrics_dataframe)

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


    # Calcular métricas desacopladas por variable
    metrics_table = get_metrics_dataframe(y_true_test, y_pred_nn, "Neural Network")

    # Guardar tabla de métricas
    os.makedirs("results/metrics", exist_ok=True)
    metrics_table.to_csv(metrics_path, index=False)

    print("Métricas guardadas en:", metrics_path)
    print("\n--- Inspección de Rangos (NARX NN) ---")
    print(f"Glucose (G)  - Real: [{y_true_test[:, 0].min():.2f}, {y_true_test[:, 0].max():.2f}] | Predicho: [{y_pred_nn[:, 0].min():.2f}, {y_pred_nn[:, 0].max():.2f}]")
    print(f"Insulina (X) - Real: [{y_true_test[:, 1].min():.2f}, {y_true_test[:, 1].max():.2f}] | Predicho: [{y_pred_nn[:, 1].min():.2f}, {y_pred_nn[:, 1].max():.2f}]")
    print(f"Insulina (I) - Real: [{y_true_test[:, 2].min():.2f}, {y_true_test[:, 2].max():.2f}] | Predicho: [{y_pred_nn[:, 2].min():.2f}, {y_pred_nn[:, 2].max():.2f}]")

    print("\n--- Métricas Enfocadas en Glucosa (G) ---")
    print(metrics_table[["model", "MAE_G", "RMSE_G", "R2_G"]].to_string(index=False))
    print("\n--- Reporte Extendido por Variable (NARX NN) ---")
    decoupled = calculate_decoupled_metrics(y_true_test, y_pred_nn)
    for var, m in decoupled.items():
        print(f"Variable {var}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")

    plot_real_vs_predicted_time(y_true_test, y_pred_nn)
    plot_real_vs_predicted_scatter(y_true_test, y_pred_nn)

    return metrics_table

def evaluate_anfis(
    test_data_path="src/data/processed/test_data.npz",
    model_path="saved_models/anfis_model.pt",
    scaler_X_path="saved_models/anfis_scaler_X.pkl",
    scaler_y_path="saved_models/anfis_scaler_y.pkl",
    metrics_path="results/metrics/anfis_metrics_eval.csv"
):
    

    print("Cargando datos de prueba para ANFIS...")

    test_data = np.load(test_data_path)
    X_test = test_data["X"]
    y_true_test = test_data["y"]

    print("Cargando scalers ANFIS...")

    with open(scaler_X_path, "rb") as f:
        scaler_X = pickle.load(f)

    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)

    X_test_scaled = scaler_X.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    input_dim = X_test.shape[1]
    output_dim = y_true_test.shape[1]

    model = ANFIS(
        input_dim=input_dim,
        output_dim=output_dim,
        num_mfs=2
    )

    print("Cargando modelo ANFIS entrenado...")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("Evaluando ANFIS...")

    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    y_pred_anfis = scaler_y.inverse_transform(y_pred_scaled)

    # Calcular métricas desacopladas por variable
    metrics_table = get_metrics_dataframe(y_true_test, y_pred_anfis, "ANFIS")

    os.makedirs("results/metrics", exist_ok=True)
    metrics_table.to_csv(metrics_path, index=False)

    print("Métricas ANFIS guardadas en:", metrics_path)
    print("\n--- Inspección de Rangos (ANFIS) ---")
    print(f"Glucose (G)  - Real: [{y_true_test[:, 0].min():.2f}, {y_true_test[:, 0].max():.2f}] | Predicho: [{y_pred_anfis[:, 0].min():.2f}, {y_pred_anfis[:, 0].max():.2f}]")
    print(f"Insulina (X) - Real: [{y_true_test[:, 1].min():.2f}, {y_true_test[:, 1].max():.2f}] | Predicho: [{y_pred_anfis[:, 1].min():.2f}, {y_pred_anfis[:, 1].max():.2f}]")
    print(f"Insulina (I) - Real: [{y_true_test[:, 2].min():.2f}, {y_true_test[:, 2].max():.2f}] | Predicho: [{y_pred_anfis[:, 2].min():.2f}, {y_pred_anfis[:, 2].max():.2f}]")

    print("\n--- Métricas Enfocadas en Glucosa (G) ---")
    print(metrics_table[["model", "MAE_G", "RMSE_G", "R2_G"]].to_string(index=False))
    print("\n--- Reporte Extendido por Variable (ANFIS) ---")
    decoupled = calculate_decoupled_metrics(y_true_test, y_pred_anfis)
    for var, m in decoupled.items():
        print(f"Variable {var}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")

    return metrics_table, y_true_test, y_pred_anfis

def compare_nn_vs_anfis():
    

    nn_metrics_path = "results/metrics/nn_metrics.csv"
    anfis_metrics_path = "results/metrics/anfis_metrics.csv"
    comparison_path = "results/metrics/model_comparison.csv"

    if not os.path.exists(nn_metrics_path):
        raise FileNotFoundError(
            "No existe nn_metrics.csv. Primero ejecuta evaluate_neural_network()."
        )

    if not os.path.exists(anfis_metrics_path):
        raise FileNotFoundError(
            "No existe anfis_metrics.csv. Primero ejecuta train_anfis() o evaluate_anfis()."
        )

    nn_metrics = pd.read_csv(nn_metrics_path)
    anfis_metrics = pd.read_csv(anfis_metrics_path)

    comparison = pd.concat([nn_metrics, anfis_metrics], ignore_index=True)

    os.makedirs("results/metrics", exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    print("Comparación guardada en:", comparison_path)
    print(comparison)

    return comparison