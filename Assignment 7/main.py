import numpy as np
import NeuralNet
import params
from util import get_data

def test_nn(X_test: np.ndarray, y_test: np.ndarray, nn: NeuralNet.FeedForwardNN):
    """
    Test the neural network on the test data

    Parameters:
        X_test (np.ndarray): The test data
        y_test (np.ndarray): The test labels
        nn (NeuralNet.FeedForwardNN): The trained neural network
    """
    errors = []
    for x, y in zip(X_test, y_test):
        y_pred = nn.forward_pass(x)
        errors.append((y - y_pred) ** 2)
    print(f"Mean squared error: {np.mean(errors)}\n")

if __name__ == "__main__":

    # Get the training and test data
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # Train the neural network
    nn = NeuralNet.FeedForwardNN(
        x_training_data=X_train, 
        y_training_data=y_train, 
        hidden_units=params.n_hidden_units, 
        learning_rate=params.learning_rate,
        activation_function=params.activation_function,
        loss_function=params.loss_function
        )
    nn.train()

    # Test the neural network
    print("Testing with trained model")
    print("----------------------------")
    print("Training data")
    test_nn(X_train, y_train, nn)
    print("Test data")
    test_nn(X_test, y_test, nn)