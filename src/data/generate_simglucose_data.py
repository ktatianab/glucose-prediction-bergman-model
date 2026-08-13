import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.basal_bolus_ctrller import BBController


def run_simglucose_simulation(patient_name='adult#001', days=3):
    """
    Ejecuta la simulación de simglucose y devuelve un DataFrame con las series temporales.
    """
    print(f"Iniciando simulación de {days} días para paciente {patient_name}...")
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Definir patrón repetitivo de comidas diarias (Desayuno, Almuerzo, Cena)
    scenario_events = []
    for day in range(days):
        day_offset = timedelta(days=day)
        scenario_events.extend([
            (day_offset + timedelta(hours=8), 45),   # Desayuno 8 AM
            (day_offset + timedelta(hours=13), 70),  # Almuerzo 1 PM
            (day_offset + timedelta(hours=20), 50)   # Cena 8 PM
        ])

    meal_scenario = CustomScenario(start_time=start_time, scenario=scenario_events)
    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName('Dexcom', seed=42)
    pump = InsulinPump.withName('Insulet')
    controller = BBController()

    env = T1DSimEnv(patient, sensor, pump, meal_scenario)
    obs, reward, done, info = env.reset()
    history = []

    end_time = start_time + timedelta(days=days)
    while not done and info['time'] < end_time:
        action = controller.policy(obs, reward, done, **info)
        obs, reward, done, info = env.step(action)

        # Estado fisiológico del paciente [Glucosa, Insulina activa, Insulina plasma]
        p_state = info['patient_state']

        history.append({
            'Time': info['time'],
            'G': obs.CGM,                  # Glucosa del sensor (mg/dL)
            'X': p_state[1] if len(p_state) > 1 else 0.0, # Insulina remota/activa
            'I': p_state[2] if len(p_state) > 2 else 0.0, # Insulina plasmática
            'u': action.basal + action.bolus,            # Insulina total inyectada (U)
            'D': info['meal']                             # Carbohidratos ingeridos (g)
        })

    return pd.DataFrame(history)


def create_dataset_matrices(df):
    """
    Transforma la serie temporal en las matrices X (k) e y (k+1) que esperan ANFIS y NARX.
    X = [G(k), X(k), I(k), u(k), D(k)]
    y = [G(k+1), X(k+1), I(k+1)]
    """
    data = df[['G', 'X', 'I', 'u', 'D']].values
    
    # X toma desde el tiempo 0 hasta N-1
    X_mat = data[:-1, :] 
    
    # y toma las variables de estado (G, X, I) desplazadas un paso al futuro: tiempo 1 hasta N
    y_mat = data[1:, :3] 

    return X_mat, y_mat


def generate_and_save_simglucose_data(days=3, test_size=0.2, seed=42):
    """
    Ejecuta el flujo completo: Simula -> Procesa matrices -> Guarda CSV y archivos .npz
    """
    df = run_simglucose_simulation(days=days)

    # 1. Guardar CSV en la carpeta data/ de la raíz
    os.makedirs("data", exist_ok=True)
    csv_path = "data/simglucose_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset crudo guardado en: {csv_path}")

    # 2. Convertir a matrices X e y
    X_mat, y_mat = create_dataset_matrices(df)

    # 3. Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y_mat, test_size=test_size, random_state=seed, shuffle=False
    )

    # 4. Guardar archivos .npz en src/data/processed/
    processed_dir = "src/data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    np.savez(os.path.join(processed_dir, "train_data.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(processed_dir, "test_data.npz"), X=X_test, y=y_test)

    print(f"Archivos procesados guardados en: {processed_dir}/train_data.npz y test_data.npz")
    return df


def plot_simulation(df):
    """
    Genera la gráfica de la simulación.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(df['Time'], df['G'], label='Sensor CGM (mg/dL)', color='tab:red')
    ax1.axhline(180, color='gray', linestyle=':', label='Límite Hiper (180)')
    ax1.axhline(70, color='blue', linestyle=':', label='Límite Hipo (70)')
    ax1.set_ylabel('Glucosa (mg/dL)')
    ax1.set_title('Simulación simglucose - Paciente Adulto #001')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.bar(df['Time'], df['D'], width=0.005, label='Comidas (g CHO)', color='tab:green', alpha=0.7)
    ax2.plot(df['Time'], df['u'], label='Insulina Total (U)', color='tab:purple', drawstyle='steps-post')
    ax2.set_ylabel('Gramos / Unidades')
    ax2.set_xlabel('Hora')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

