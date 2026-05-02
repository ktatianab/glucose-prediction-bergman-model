


from src.data.generate_data import save_data, simulate_one
from src.training.train_neural_network import train_neural_network
from src.evaluation.evaluate_models import evaluate_neural_network



def main():
    
    # Definición de parámetros
    p1 = 0.028
    p2 = 0.025
    p3 = 5e-5
    p4 = 0.05
    Gb = 90.0
    Ib = 15.0

    '''
    # Generación de datos
    print("Generación de datos")
    save_data(p1, p2, p3, p4, Gb, Ib, num_samples=100)

    '''

    #Entrenamiento de la red neuronal
    print("Entrenamiento de la red neuronal")
    train_neural_network(100, 64, 0.001) #epochs, batch_size, learning_rate

    print("Evaluación de la red neuronal")
    evaluate_neural_network()
    



if __name__ == "__main__":
    main()
    