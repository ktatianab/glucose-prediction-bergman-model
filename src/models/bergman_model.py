import numpy as np

'''
 En este archivo se implementa el modelo de Bergman,
 un modelo matematico que describe la dinamica de la glucosa e insulina en el cuerpo humano. 
 
'''

'''Funcion con la dinamica del modelo de Bergman'''
def bergman_derivatives(state, t, p1, p2, p3, p4, Gb, Ib, u):
    G, X, I = state
    u_in = u

    # Ecuaciones del modelo minimo de Bergman
    dGdt = -p1*(G - Gb) - G*X
    dXdt = -p2*X + p3*(I - Ib)
    dIdt = -p4*(I - Ib) + u_in

    return np.array([dGdt, dXdt, dIdt])


''' Funcion para simular un paso del modelo de Bergman usando Euler (discretizacion)'''

def bergman_step(state, t, params, Ts, u):
    p1, p2, p3, p4, Gb, Ib = params

    derivatives = bergman_derivatives( state, t, p1, p2, p3, p4, Gb, Ib, u)

    next_state = state + Ts * derivatives
    return next_state