import numpy as np
import jax.numpy as jnp

def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2

def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test

def loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    The mean squared error loss function.
    """
    return jnp.mean((y_true - y_pred) ** 2)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    The sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    """
    The ReLU activation function.
    """
    return np.maximum(0, x)

def tanh(x: np.ndarray) -> np.ndarray:
    """
    The tanh activation function.
    """
    return np.tanh(x)

def softplus(x: np.ndarray) -> np.ndarray:
    """
    The softplus activation function.
    """
    return np.log(1 + np.exp(x))