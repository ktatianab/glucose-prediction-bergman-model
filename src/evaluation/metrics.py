import numpy as np
import pandas as pd
from sklearn.metrics import ( mean_absolute_error, mean_squared_error, r2_score)

'''
En este archivo se definen las funciones para calcular las métricas de evaluación de los modelos, 
como el error absoluto medio (MAE), 
el error cuadrático medio (MSE), 
el error cuadrático medio (RMSE) 
y el coeficiente de determinación R². 
Se añade soporte para desacoplar métricas por variable (G, X, I).
'''

# Función para calcular el error absoluto medio (MAE)
def calculate_mae(y_true, y_pred, multioutput='uniform_average'):
    return mean_absolute_error(y_true, y_pred, multioutput=multioutput)

# Función para calcular el error cuadrático medio (MSE)
def calculate_mse(y_true, y_pred, multioutput='uniform_average'):
    return mean_squared_error(y_true, y_pred, multioutput=multioutput)

# Función para calcular la raíz del error cuadrático medio (RMSE)
def calculate_rmse(y_true, y_pred, multioutput='uniform_average'):
    if multioutput == 'raw_values':
        mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
        return np.sqrt(mse)
    else:
        mse = mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)

# Función para calcular el coeficiente de determinación R²
def calculate_r2(y_true, y_pred, multioutput='uniform_average'):
    return r2_score(y_true, y_pred, multioutput=multioutput)

# Función para calcular todas las métricas globales promediadas
def calculate_all_metrics(y_true, y_pred):
    metrics = {
        "MAE": calculate_mae(y_true, y_pred),
        "MSE": calculate_mse(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "R2": calculate_r2(y_true, y_pred)
    }
    return metrics

# Nueva función para calcular métricas desacopladas por variable
def calculate_decoupled_metrics(y_true, y_pred, variables=["G", "X", "I"]):
    results = {}
    for i, var in enumerate(variables):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        results[var] = {
            "MAE": mean_absolute_error(yt, yp),
            "MSE": mean_squared_error(yt, yp),
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "R2": r2_score(yt, yp)
        }
    return results

# Nueva función para obtener un DataFrame formateado con métricas desacopladas
def get_metrics_dataframe(y_true, y_pred, model_name, variables=["G", "X", "I"]):
    decoupled = calculate_decoupled_metrics(y_true, y_pred, variables=variables)
    row = {"model": model_name}
    for var, metrics in decoupled.items():
        row[f"MAE_{var}"] = metrics["MAE"]
        row[f"MSE_{var}"] = metrics["MSE"]
        row[f"RMSE_{var}"] = metrics["RMSE"]
        row[f"R2_{var}"] = metrics["R2"]
    return pd.DataFrame([row])