import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.models.bergman_model import bergman_step
from src.utils import set_seed


def generate_data(num_simulations, time_steps, Ts, p1, p2, p3, p4, Gb, Ib):
    set_seed(42)

    X_data = []
    y_data = []
   

    params = [p1, p2, p3, p4, Gb, Ib]

    for sim in range(num_simulations):

       
        G0 = np.random.uniform(70, 180)
        X0 = np.random.uniform(0, 0.05)
        I0 = np.random.uniform(5, 30)

        state = np.array([G0, X0, I0], dtype=float)

        for k in range(time_steps):
            t = k * Ts

            # insulin input u(k)
            u = 1.0 if 20 <= t < 40 else 0.0

            # Perturbation D(k)
            D = 10.0 if t == 50 else 0.0

            # input: [G, X, I, u, D]
            input_k = np.array([state[0], state[1], state[2], u, D])

            # output: [G_next, X_next, I_next]
            next_state = bergman_step(state, t, params, Ts, u)

            X_data.append(input_k)
            y_data.append(next_state)

            state = next_state

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data


def save_data(num_simulations, time_steps, Ts, p1, p2, p3, p4, Gb, Ib):

    X, y = generate_data(num_simulations, time_steps, Ts, p1, p2, p3, p4, Gb, Ib)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    os.makedirs("src/data/processed", exist_ok=True)

    np.savez("src/data/processed/train_data.npz", X=X_train, y=y_train)
    np.savez("src/data/processed/test_data.npz", X=X_test, y=y_test)

    return X_train, X_test, y_train, y_test

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