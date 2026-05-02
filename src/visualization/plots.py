import os
import numpy as np
import matplotlib.pyplot as plt


'''
    En este archivo se definen funciones para graficar los resultados de las simulaciones
    y el entrenamiento de la red neuronal.

'''

# Funcion que grafica la simulacion  del modelo de Bergman

def plot_simulation(t, states, u, D):

    G = states[:, 0]
    X = states[:, 1]
    I = states[:, 2]

    #Crea figura con 5 graficas
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(t, G)
    axs[0].set_ylabel("G(t)")
    axs[0].set_title("Glucosa G(t)")
    axs[0].grid(True)

    axs[1].plot(t, X)
    axs[1].set_ylabel("X(t)")
    axs[1].set_title("Acción remota de insulina X(t)")
    axs[1].grid(True)

    axs[2].plot(t, I)
    axs[2].set_ylabel("I(t)")
    axs[2].set_title("Insulina I(t)")
    axs[2].grid(True)

    axs[3].plot(t, u)
    axs[3].set_ylabel("u(t)")
    axs[3].set_title("Entrada de insulina u(t)")
    axs[3].grid(True)

    axs[4].plot(t, D)
    axs[4].set_ylabel("D(t)")
    axs[4].set_xlabel("Tiempo [min]")
    axs[4].set_title("Perturbación D(t)")
    axs[4].grid(True)

    plt.tight_layout()

    #Guarda la figura
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(
        "results/figures/01_bergman_simulation.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    return fig


"""
    Funcion que grafica cómo cambia el error o pérdida durante el entrenamiento de la red neuronal.

    Espera un diccionario como:
        history = {
            "train_loss": [...],
            "test_loss": [...]
        }

 """

def plot_training_loss(history):
    

    os.makedirs("results/figures", exist_ok=True) #Crea el directorio si no existe

    train_loss = history["train_loss"]
    test_loss = history["test_loss"]

    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    #Dibuja 2 curvas de pérdida, la de entrenamiento y la prueba
    ax.plot(epochs, train_loss, label="Training loss") #Grafica el loss de entrenamiento, deberia bajar
    ax.plot(epochs, test_loss, label="Test loss")#Grafica el loss de test, deber bajar o mantenerse estable


    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Evolución del loss durante el entrenamiento")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    fig.savefig(
        "results/figures/02_nn_training_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    return fig

# Función que grafica la comparación entre los valores reales y los predichos por la red neuronal

def plot_real_vs_predicted_time(y_true, y_pred, t=None):
    

    os.makedirs("results/figures", exist_ok=True) #Crea el directorio si no existe

    if t is None:
        t = np.arange(y_true.shape[0])

    variable_names = ["G", "X", "I"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)


    #Crea 3 graficas, una para cada variable (G, X, I) 
    for i in range(3):

        axs[i].plot(t, y_true[:, i], label=f"{variable_names[i]} real")
        axs[i].plot(t, y_pred[:, i], "--", label=f"{variable_names[i]} predicho")

        axs[i].set_ylabel(variable_names[i])
        axs[i].set_title(f"{variable_names[i]} real vs predicho")
        axs[i].grid(True)
        axs[i].legend()


    axs[-1].set_xlabel("Tiempo / muestra")

    plt.tight_layout()

    #Guarda la figura
    fig.savefig( "results/figures/03_nn_prediction.png", dpi=300, bbox_inches="tight")

    plt.show()

    return fig



"""
    Funcion que Grafica scatter real vs predicho para:
        G_next, X_next, I_next

    Si la red predice perfectamente, los puntos deberían caer cerca
    de la línea diagonal.

"""

def plot_real_vs_predicted_scatter(y_true, y_pred):
    
    #Crea el directorio si no existe
    os.makedirs("results/figures", exist_ok=True)

    variable_names = ["G", "X", "I"]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    for i in range(3):
        axs[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.6)

        min_value = min(y_true[:, i].min(), y_pred[:, i].min())
        max_value = max(y_true[:, i].max(), y_pred[:, i].max())

        axs[i].plot(
            [min_value, max_value],
            [min_value, max_value],
            "--"
        )

        axs[i].set_xlabel(f"{variable_names[i]} real")
        axs[i].set_ylabel(f"{variable_names[i]} predicho")
        axs[i].set_title(f"Scatter {variable_names[i]}")
        axs[i].grid(True)

    plt.tight_layout()

    fig.savefig(
        "results/figures/04_nn_scatter.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    return fig