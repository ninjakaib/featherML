import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self):
        self.weights = None
        self.loss_history = []  # To store the loss at each epoch

    def fit(self, X, y, closed_form_solver=True, learning_rate=0.01, epochs=1000):
        
        # Add a bias term (1) to the input features
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        
        if closed_form_solver:
            # Compute weights using the normal equation
            self.weights = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)

        else:
            # Initialize weights
            self.weights = np.random.randn(X_bias.shape[1], 1)
            
            # Gradient Descent
            for epoch in range(epochs):
                gradients = -2/X.shape[0] * X_bias.T.dot(y - X_bias.dot(self.weights))
                self.weights -= learning_rate * gradients
                
                # Compute the loss (MSE) and store it
                y_pred = X_bias.dot(self.weights)
                mse = np.mean((y - y_pred) ** 2)
                self.loss_history.append(mse)

            # Plot the loss over epochs
            self._plot_loss()

        return self
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("The model has not been trained yet.")
        
        # Add a bias term (1) to the input features
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        
        return X_bias.dot(self.weights)

    def evaluate(self, X, y):
        # Mean Squared Error (MSE) as the evaluation metric
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse

    def _plot_loss(self):
        """Helper function to plot the loss over epochs."""
        plt.plot(self.loss_history)
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.show()
