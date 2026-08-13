import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo para evitar bloqueos por plt.show() en ejecución desasistida
from src.data.generate_simglucose_data import generate_and_save_simglucose_data
from src.training.train_neural_network import train_neural_network
from src.evaluation.evaluate_models import evaluate_neural_network, evaluate_anfis, compare_nn_vs_anfis
from src.training.train_anfis import train_anfis
from src.utils import set_seed

'''
    Este es el archivo principal que se ejecuta para generar los datos, entrenar los modelos y evaluar su desempeño.
    Se llama a las funciones de generación de datos, entrenamiento y evaluación en el orden correcto.
'''

def main():

    set_seed(42)
    
    # Generación de datos con simglucose (3 días)
    print("Generación de datos con simglucose...")
    generate_and_save_simglucose_data(days=3, test_size=0.2, seed=42)
 
    # Entrenamiento de la red neuronal
    print("Entrenamiento de la red neuronal")
    train_neural_network(100, 64, 0.001) #epochs, batch_size, learning_rate

    print("Evaluación de la red neuronal")
    evaluate_neural_network()
    
    # Entrenamiento de ANFIS
    print("Entrenamiento ANFIS")
    train_anfis(
        epochs=300,
        batch_size=64,
        learning_rate=0.001,
        num_mfs=2,
        seed=42
    )
    print("Evaluación ANFIS")
    evaluate_anfis()

    print("Comparando red neuronal vs ANFIS...")
    compare_nn_vs_anfis()


if __name__ == "__main__":
    main()

    