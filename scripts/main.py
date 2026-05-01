
from src.visualization.plots import plot_simulation
from src.data.generate_data import save_data, simulate_one
import numpy as np
from src.data.generate_data import save_data, simulate_one
from src.visualization.plots import plot_simulation
from src.training.train_neural_network import train_neural_network



def main():
    
   
    # Bergman's Minimal Model parameters
    p1 = 0.028
    p2 = 0.025
    p3 = 5e-5
    p4 = 0.05
    Gb = 90.0
    Ib = 15.0

    #Neural network training
    print("Neural network training")
    train_neural_network()





if __name__ == "__main__":
    main()
    