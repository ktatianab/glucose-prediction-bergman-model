
import torch.nn as nn

'''
    En este archivo se define la arquitectura de la red neuronal NARXNetwork, 
    que es una red feedforward simple con 2 capas ocultas. Esta red se utiliza para modelar la relación entre las entradas (glucosa, insulina, carbohidratos, etc.) y las salidas (glucosa futura) en el sistema de Bergman.

'''
class NARXNetwork(nn.Module):
    def __init__(self, input_dim=5, output_dim=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)