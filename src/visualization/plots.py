
import os
import matplotlib.pyplot as plt


def plot_simulation(t, states, u, D):
    """
    Genera una figura con:
    G(t), X(t), I(t), u(t), D(t)
    """

    G = states[:, 0]
    X = states[:, 1]
    I = states[:, 2]

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

    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/01_bergman_simulation.png", dpi=300, bbox_inches="tight")


    plt.show()

    return fig