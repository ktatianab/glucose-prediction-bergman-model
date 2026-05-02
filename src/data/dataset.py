
import torch
from torch.utils.data import Dataset

'''
 En este archivo se define la clase BergmanDataset, que es una subclase de Dataset de PyTorch.
 Esta clase se utiliza para cargar los datos generados en el archivo generate_data.py, y prepararlos
 para su uso en el entrenamiento y evaluación de la red neuronal.
 
'''

class BergmanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]