import torch
import torch.nn as nn

''' 

 Este archivo se define la clase ANFIS, que implementa un sistema 
 de inferencia difusa adaptativo de Takagi-Sugeno de primer orden.

'''

'''
    Capa de funciones de membresía gaussianas.

    Para cada entrada x_i se crean varias funciones gaussianas:
        mu_ij(x_i) = exp(-0.5 * ((x_i - c_ij) / sigma_ij)^2)

    input_dim: número de entradas del sistema.
    num_mfs: número de funciones de membresía por entrada.
  
'''
class GaussianMembership(nn.Module):

    def __init__(self, input_dim, num_mfs):
        super().__init__()

        self.input_dim = input_dim
        self.num_mfs = num_mfs

        # Centros iniciales distribuidos entre -1 y 1.
        centers = torch.linspace(-1.0, 1.0, num_mfs).repeat(input_dim, 1)

        # Sigmas iniciales positivos.
        sigmas = torch.ones(input_dim, num_mfs)

        self.centers = nn.Parameter(centers)
        self.log_sigmas = nn.Parameter(torch.log(sigmas))

    def forward(self, x):

        x = x.unsqueeze(2)  # [batch_size, input_dim, 1]
        sigmas = torch.exp(self.log_sigmas) + 1e-6

        membership_values = torch.exp(
            -0.5 * ((x - self.centers) / sigmas) ** 2
        )

        return membership_values



'''
ANFIS Takagi-Sugeno de primer orden.

Entrada:
    x = [G(k), X(k), I(k), u(k), D(k)]

Salida:
    y = [G(k+1), X(k+1), I(k+1)]

Estructura:
    - Funciones de membresía gaussianas.
    - Reglas fuzzy por combinación de membresías.
    - Consecuentes lineales por regla.
    - Salida como promedio ponderado normalizado.
'''

class ANFIS(nn.Module):


    def __init__(self, input_dim=5, output_dim=3, num_mfs=2):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_mfs = num_mfs
        self.num_rules = num_mfs ** input_dim

        self.membership = GaussianMembership(input_dim, num_mfs)

        # Cada regla tiene un consecuente lineal:
        # y_r = a_1*x_1 + ... + a_n*x_n + b
        #
        # consequent_weights:
        # [num_rules, output_dim, input_dim + 1]
        # El +1 corresponde al sesgo.
        self.consequent_weights = nn.Parameter(
            0.01 * torch.randn(self.num_rules, output_dim, input_dim + 1)
        )

        # Índices que indican qué función de membresía usa cada regla.
        # Para input_dim=5 y num_mfs=2, se generan 32 reglas.
        rule_indices = torch.cartesian_prod(
            *[torch.arange(num_mfs) for _ in range(input_dim)]
        )

        self.register_buffer("rule_indices", rule_indices)

    def compute_rule_strengths(self, membership_values):
        """
        membership_values:
            [batch_size, input_dim, num_mfs]

        Retorna:
            normalized_strengths: [batch_size, num_rules]
        """

        batch_size = membership_values.shape[0]

        # Para cada regla se toman las membresías correspondientes
        # a cada entrada.
        rule_strengths = []

        for rule in self.rule_indices:
            selected_mfs = membership_values[
                torch.arange(batch_size).unsqueeze(1),
                torch.arange(self.input_dim).unsqueeze(0),
                rule.unsqueeze(0)
            ]

            # Producto de membresías para obtener fuerza de disparo.
            strength = torch.prod(selected_mfs, dim=1)
            rule_strengths.append(strength)

        rule_strengths = torch.stack(rule_strengths, dim=1)

        # Normalización de las fuerzas de disparo.
        normalized_strengths = rule_strengths / (
            torch.sum(rule_strengths, dim=1, keepdim=True) + 1e-8
        )

        return normalized_strengths

    def forward(self, x):
        """
        x: [batch_size, input_dim]

        Retorna:
        y_pred: [batch_size, output_dim]
        """

        batch_size = x.shape[0]

        membership_values = self.membership(x)
        normalized_strengths = self.compute_rule_strengths(membership_values)

        # Agregar término de sesgo.
        ones = torch.ones(batch_size, 1, device=x.device)
        x_augmented = torch.cat([x, ones], dim=1)

        # Consecuentes lineales para cada regla.
        #
        # x_augmented: [batch_size, input_dim + 1]
        # consequent_weights: [num_rules, output_dim, input_dim + 1]
        #
        # rule_outputs: [batch_size, num_rules, output_dim]
        rule_outputs = torch.einsum(
            "bi,roi->bro",
            x_augmented,
            self.consequent_weights
        )

        # Promedio ponderado por fuerza normalizada de cada regla.
        y_pred = torch.sum(
            normalized_strengths.unsqueeze(2) * rule_outputs,
            dim=1
        )

        return y_pred