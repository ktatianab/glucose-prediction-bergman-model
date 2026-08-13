import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.models.anfis_model import ANFIS
from src.evaluation.metrics import (calculate_all_metrics, calculate_decoupled_metrics, get_metrics_dataframe)

'''
Este archivo contiene la función de entrenamiento del modelo ANFIS.
Se encarga de cargar los datos, normalizarlos, definir el modelo, entrenarlo    
y guardar el modelo entrenado junto con los scalers y el historial de entrenamiento.

'''


def set_torch_seed(seed=42):
    """
    Fija semillas para reproducibilidad.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
    Entrena un modelo ANFIS básico para identificar la dinámica discretizada
    del modelo de Bergman.

    Entrada:
        X = [G(k), X(k), I(k), u(k), D(k)]

    Salida:
        y = [G(k+1), X(k+1), I(k+1)]
"""

def train_anfis(
    epochs=300,
    batch_size=64,
    learning_rate=0.001,
    num_mfs=2,
    seed=42
):
    

    set_torch_seed(seed)

    train_path = "src/data/processed/train_data.npz"
    test_path = "src/data/processed/test_data.npz"

    model_path = "saved_models/anfis_model.pt"
    scaler_X_path = "saved_models/anfis_scaler_X.pkl"
    scaler_y_path = "saved_models/anfis_scaler_y.pkl"
    history_path = "results/metrics/anfis_history.npy"
    metrics_path = "results/metrics/anfis_metrics.csv"

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    print("Cargando datos para ANFIS...")

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]

    print("Tamaño X_train:", X_train.shape)
    print("Tamaño y_train:", y_train.shape)
    print("Tamaño X_test:", X_test.shape)
    print("Tamaño y_test:", y_test.shape)

    print("Normalizando datos para ANFIS...")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    input_dim = X_train_tensor.shape[1]
    output_dim = y_train_tensor.shape[1]

    print("Entradas ANFIS:", input_dim)
    print("Salidas ANFIS:", output_dim)
    print("Funciones de membresía por entrada:", num_mfs)
    print("Número de reglas:", num_mfs ** input_dim)

    model = ANFIS(
        input_dim=input_dim,
        output_dim=output_dim,
        num_mfs=num_mfs
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "test_loss": []
    }

    print("Entrenando ANFIS...")

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor)
            test_loss = loss_fn(test_pred, y_test_tensor).item()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss:.6f} "
                f"Test Loss: {test_loss:.6f}"
            )

    print("Guardando modelo ANFIS...")

    torch.save(model.state_dict(), model_path)

    with open(scaler_X_path, "wb") as f:
        pickle.dump(scaler_X, f)

    with open(scaler_y_path, "wb") as f:
        pickle.dump(scaler_y, f)

    np.save(history_path, history)

    print("Modelo ANFIS guardado en:", model_path)
    print("Scaler X ANFIS guardado en:", scaler_X_path)
    print("Scaler y ANFIS guardado en:", scaler_y_path)
    print("Historial ANFIS guardado en:", history_path)

    print("Evaluando ANFIS en test...")

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_real = scaler_y.inverse_transform(y_test_scaled)

    metrics_table = get_metrics_dataframe(y_real, y_pred, "ANFIS")

    print("Métricas finales ANFIS (Desacopladas):")
    print(metrics_table[["model", "MAE_G", "RMSE_G", "R2_G"]].to_string(index=False))
    print("\nReporte Extendido por Variable (ANFIS):")
    decoupled = calculate_decoupled_metrics(y_real, y_pred)
    for var, m in decoupled.items():
        print(f"Variable {var}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")

    metrics_table.to_csv(metrics_path, index=False)
    print("\nMétricas ANFIS guardadas en:", metrics_path)

    return model, scaler_X, scaler_y, history