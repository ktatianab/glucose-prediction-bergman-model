import numpy as np
from sklearn.metrics import ( mean_absolute_error, mean_squared_error, r2_score)

'''
En este archivo se definen las funciones para calcular las métricas de evaluación de los modelos, 
como el error absoluto medio (MAE), 
el error cuadrático medio (MSE), 
la raíz del error cuadrático medio (RMSE) 
y el coeficiente de determinación R². 

'''


#Función para calcular el error absoluto medio (MAE)
def calculate_mae(y_true, y_pred):
    
    return mean_absolute_error(y_true, y_pred)

#Función para calcular el error cuadrático medio (MSE)
def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

#Función para calcular la raíz del error cuadrático medio (RMSE)
def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

#Función para calcular el coeficiente de determinación R²
def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

#Función para calcular todas las métricas
def calculate_all_metrics(y_true, y_pred):
    metrics = {
        "MAE": calculate_mae(y_true, y_pred),
        "MSE": calculate_mse(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "R2": calculate_r2(y_true, y_pred)
    }
    return metrics