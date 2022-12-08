import numpy as np
import warnings
import matplotlib.pyplot as plt


class BaseRegressor:
    """
    A base class for regression models.
    """
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fits the model to the data.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement this method")
        return X @ self.weights + self.bias

    def score(self, X, y):
        """
        Returns the mean squared error of the model.
        """
        return np.mean((self.predict(X) - y)**2)




class LinearRegressor(BaseRegressor):
    
    def __init__(self, method="global", regularization="ridge", regstrength=0.1, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.regularization = regularization
        self.regstrength = regstrength

    def _fit_global(self, X, y):
        
        if self.regularization is None:
            self.weights = np.linalg.inv(X.T@X) @ X@y
        elif self.regularization == 'ridge':
            self.weights = np.linalg.inv(X.T@x - np.eye(np.shape(X)[0]*self.regstrength))@X.T@y
        else:
            warnings.warn('Choose between ridge and None; defaulting to None')
            self.weights = np.linalg.inv(X.T@X) @ X@y
        self.bias = np.mean(y-X@self.weights)

    def _fit_iterative(self, X, y, learning_rate=0.01):
        
        for i in range(X.shape[0]):
            self.weights += learning_rate * (y[i] - X[i] @ self.weights - self.bias) * X[i] - self.regstrength * self.weights
        self.weights /= X.shape[0]
        return self.weights, self.bias

    def fit(self, X, y):
        
        if method == 'global':
            out = self.fit_global(X,y)
        elif method == 'iterative':
            out = self.fit_iterative(X,y)
        else:
            warnings.warn('method not recognised, defaulting to global fit')
            out = self.fit_global
        return out
    
    def featurize_flowfield(field):
    """
    Compute features of a 2D spatial field. These features are chosen based on the 
    intuition that the input field is a 2D spatial field with time translation 
    invariance.
    The output is an augmented feature along the last axis of the input field.
    Args:
        field (np.ndarray): A 3D array of shape (batch, nx, ny) containing the flow field
    Returns:
        field_features (np.ndarray): A 3D array of shape (batch, nx, ny, M) containing 
            the computed features stacked along the last axis
    """

    ## Compute the Fourier features
    field_fft = np.fft.fftshift(np.fft.fft2(field))
    field_fft_abs = np.log(np.abs(field_fft) + 1e-8)[..., None]
    field_fft_phase = np.angle(field_fft)[..., None]

    ## Compute the spatial gradients along x and y
    field_grad = np.stack(np.gradient(field, axis=(-2, -1)), axis=-1)

    ## Compute the spatial Laplacian
    field_lap = np.sum(np.stack(np.gradient(field_grad, axis=(-2, -1)), axis=-1), axis=-1)

    field = field[..., None]
    field_features = np.concatenate(
        [field, field_grad, field_lap, field_fft_phase, field_fft_abs], 
        axis=-1
    )
    return field_features
