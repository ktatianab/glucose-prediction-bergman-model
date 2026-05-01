import numpy as np

# Bergman’s Minimal Model of blood glucose dynamics
def bergman_derivatives(state, t, p1, p2, p3, p4, Gb, Ib, u):
    G, X, I = state
    u_in = u

    # Bergman model equations
    dGdt = -p1*(G - Gb) - G*X
    dXdt = -p2*X + p3*(I - Ib)
    dIdt = -p4*(I - Ib) + u_in

    return np.array([dGdt, dXdt, dIdt])


# Function to simulate the discretized Bergman model using the Euler method
def bergman_step(state, t, params, Ts, u):
    p1, p2, p3, p4, Gb, Ib = params

    derivatives = bergman_derivatives(
        state, t, p1, p2, p3, p4, Gb, Ib, u
    )

    next_state = state + Ts * derivatives
    return next_state