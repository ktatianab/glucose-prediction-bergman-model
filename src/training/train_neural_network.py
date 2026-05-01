
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.models.neural_network import NARXNetwork


def train_neural_network():
    

    epochs = 200
    batch_size = 64
    learning_rate = 0.001

    train_path = "src/data/processed/train_data.npz"
    test_path = "src/data/processed/test_data.npz"

    model_path = "saved_models/neural_network.pt"
    scaler_X_path = "saved_models/scaler_X.pkl"
    scaler_y_path = "saved_models/scaler_y.pkl"
    history_path = "results/metrics/nn_history.npy"

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    print("Cargando datos...")

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

    print("Normalizando datos...")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    input_size = X_train_tensor.shape[1]
    output_size = y_train_tensor.shape[1]

    print("Entradas de la red:", input_size)
    print("Salidas de la red:", output_size)

    model = NARXNetwork(
        input_dim=input_size,
        output_dim=output_size
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "test_loss": []
    }

    print("Entrenando red neuronal...")

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        model.eval()

        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = loss_fn(test_outputs, y_test_tensor).item()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss:.6f} "
                f"Test Loss: {test_loss:.6f}"
            )

    torch.save(model.state_dict(), model_path)

    with open(scaler_X_path, "wb") as f:
        pickle.dump(scaler_X, f)

    with open(scaler_y_path, "wb") as f:
        pickle.dump(scaler_y, f)

    np.save(history_path, history)

    print("Entrenamiento finalizado.")
    print("Modelo guardado en:", model_path)
    print("Scaler X guardado en:", scaler_X_path)
    print("Scaler y guardado en:", scaler_y_path)
    print("Historial guardado en:", history_path)

    return model, scaler_X, scaler_y, history