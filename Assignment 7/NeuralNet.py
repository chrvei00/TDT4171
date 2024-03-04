import numpy as np
import jax.numpy as jnp
from jax import value_and_grad

class FeedForwardNN:
    def __init__(self, x_training_data: np.ndarray, y_training_data: np.ndarray, hidden_units: int = 10, output_units: int = 1, learning_rate: float = 0.01, activation_function: callable = lambda x: 1 / (1 + np.exp(-x)), loss_function: callable = lambda y_true, y_pred: jnp.mean((y_true - y_pred) ** 2)):
        self.x_training_data = x_training_data
        self.y_training_data = y_training_data
        self.training_data = list(zip(x_training_data, y_training_data))
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.learning_rate = learning_rate
        self.input_units = x_training_data.shape[1]
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.weights = self.initialize_weights(self.input_units, hidden_units, output_units)
        self.biases = self.initialize_biases(hidden_units, output_units)
        self.grad_loss = value_and_grad(self.loss_function, argnums=1)
                    
    def __str__(self) -> str:
        return f"Weights: {self.weights}\nBiases: {self.biases}"
    
    def initialize_weights(self, input_units: int, hidden_units: int, output_units: int) -> list[np.ndarray]:
        """Initializes the weights of the neural network"""
        weights = [np.random.rand(input_units, hidden_units)]
        weights.append(np.random.rand(hidden_units, output_units))
        return weights

    def initialize_biases(self, hidden_units: int, output_units: int) -> list[np.ndarray]:
        """Initializes the biases of the neural network"""
        biases = [np.random.rand(hidden_units)]
        biases.append(np.random.rand(output_units))
        return biases

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        """The forward pass of the neural network"""
        hidden_layer = self.activation_function(np.dot(x, self.weights[0]) + self.biases[0])
        output_layer = self.activation_function(np.dot(hidden_layer, self.weights[1]) + self.biases[1])
        return output_layer
    
    def backward_pass(self) -> float:
        """The backward pass of the neural network"""
        # Get the error and the gradient
        error, grad = self.grad_loss(self.y_training_data, self.forward_pass(self.x_training_data))
        # Update the weights
        self.weights[0] -= self.learning_rate * grad[0]
        self.weights[1] -= self.learning_rate * grad[1]
        # Update the biases
        self.biases[0] -= self.learning_rate * grad[2]
        self.biases[1] -= self.learning_rate * grad[3]
        # Return the error
        return error
        
    def train(self):
        """Trains the neural network"""
        # print(f"\nInital Weights: \n{self.weights[0]}")
        # print(f"Inital Biases: \n{self.biases[0]}")
        errors = []
        for x, y in self.training_data:
            # FeedForward
            errors.append(self.backward_pass())
        # print(f"\nFinal Weights: \n{self.weights[0]}")
        # print(f"Final Biases: \n{self.biases[0]}")
        print(f"\nMean squared error (training): {np.mean(errors)}\n")