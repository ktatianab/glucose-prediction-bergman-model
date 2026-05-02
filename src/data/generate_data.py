

import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.models.bergman_model import bergman_step
from src.utils import set_seed


'''
En este archivo, se genera un conjunto de datos para el entrenamiento y evaluación del modelo. 
Simula múltiples trayectorias del sistema de Bergman con diferentes condiciones iniciales y entradas,
y guarda los datos en archivos .npz para su uso posterior.

'''

'''

Función para generación de datos para entrenamiento y evaluación:

Con parametros:
    num_simulations: cuántas simulaciones independientes se van a generar.
    time_steps: cuántos pasos de tiempo tiene cada simulación.
    Ts: tiempo de muestreo.
    p1, p2, p3, p4: parámetros del modelo de Bergman.
    Gb: glucosa basal.
    Ib: insulina basal

'''

def generate_data(num_simulations, time_steps, Ts, p1, p2, p3, p4, Gb, Ib):
    

    set_seed(42) #semilla aleatoria para reproducibilidad (obtener los mismos datos cada vez que se ejecute)

    X_data = [] 
    y_data = []
   

    params = [p1, p2, p3, p4, Gb, Ib]

    
    for sim in range(num_simulations):

       #Generar condiciones iniciales aleatorias para G, X e I dentro de rangos fisiológicos típicos.

        G0 = np.random.uniform(70, 180)
        X0 = np.random.uniform(0, 0.05)
        I0 = np.random.uniform(5, 30)

        state = np.array([G0, X0, I0], dtype=float)


        for k in range(time_steps):
            t = k * Ts

            # insulin input u(k), entre 20 y 40 minutos, y es cero en otros momentos.
            u = 1.0 if 20 <= t < 40 else 0.0

            # Perturbation D(k), por comida o carga de glucosa, en el minuto 50, y es cero en otros momentos.
            D = 10.0 if t == 50 else 0.0

            # input: [G, X, I, u, D], en el instante actual
            input_k = np.array([state[0], state[1], state[2], u, D])

            # output: [G_next, X_next, I_next], usando el modelo de bergman
            next_state = bergman_step(state, t, params, Ts, u, D)

            X_data.append(input_k)
            y_data.append(next_state)

            state = next_state

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data

'''
Función para guardar los datos generados en archivos .npz, y dividirlos en conjuntos de entrenamiento y prueba.

'''
def save_data(num_simulations, time_steps, Ts, p1, p2, p3, p4, Gb, Ib):

    # Generar los datos
    X, y = generate_data(num_simulations, time_steps, Ts, p1, p2, p3, p4, Gb, Ib)

    #Divide entre datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2, #20% de los datos para prueba, 80% para entrenamiento
        random_state=42, #semilla aleatoria para reproducibilidad
        shuffle=True #Mezclar los datos
    )


    os.makedirs("src/data/processed", exist_ok=True) #Crear el directorio si no existe
    np.savez("src/data/processed/train_data.npz", X=X_train, y=y_train) #Guardar los datos de entrenamiento en un archivo .npz
    np.savez("src/data/processed/test_data.npz", X=X_test, y=y_test) #Guardar los datos de prueba en un archivo .npz

    print(f"Datos guardados en src/data/processed/train_data.npz y src/data/processed/test_data.npz")
    return X_train, X_test, y_train, y_test


'''Función para simular una sola trayectoria del sistema de Bergman,
    con condiciones iniciales aleatorias y entradas específicas.'''

def simulate_one(time_steps, Ts, p1, p2, p3, p4, Gb, Ib):
    params = [p1, p2, p3, p4, Gb, Ib]

    G0 = np.random.uniform(70, 180)
    X0 = np.random.uniform(0, 0.05)
    I0 = np.random.uniform(5, 30)

    state = np.array([G0, X0, I0], dtype=float)

    t_values = []
    states = []
    u_values = []
    D_values = []

    for k in range(time_steps):
        t = k * Ts

        u = 1.0 if 20 <= t < 40 else 0.0
        D = 10.0 if t == 50 else 0.0

        t_values.append(t)
        states.append(state.copy())
        u_values.append(u)
        D_values.append(D)

        next_state = bergman_step(state, t, params, Ts, u)
        state = next_state

    return (
        np.array(t_values),
        np.array(states),
        np.array(u_values),
        np.array(D_values)
    )