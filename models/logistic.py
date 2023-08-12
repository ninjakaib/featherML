import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate=0.01, epochs=1000):
        super().__init__()
        self.weights = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Add a bias term (1) to the input features
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Initialize weights
        self.weights = np.random.randn(X_bias.shape[1], 1)
        
        # Gradient Descent
        for epoch in range(self.epochs):
            logits = X_bias.dot(self.weights)
            predictions = self._sigmoid(logits)
            error = y - predictions
            gradients = X_bias.T.dot(error)
            self.weights += self.learning_rate * gradients

    def predict_proba(self, X):
        """Predict probabilities."""
        if self.weights is None:
            raise ValueError("The model has not been trained yet.")
        
        # Add a bias term (1) to the input features
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        logits = X_bias.dot(self.weights)
        
        return self._sigmoid(logits)

    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X, y):
        """Compute the log loss."""
        y_pred_proba = self.predict_proba(X)
        epsilon = 1e-15  # to avoid log(0)
        log_loss = -np.mean(y * np.log(y_pred_proba + epsilon) + (1 - y) * np.log(1 - y_pred_proba + epsilon))
        return log_loss
