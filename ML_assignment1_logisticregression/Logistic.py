import numpy as np

class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        """
        Parameters:
        - penalty: str, "l1" or "l2". Determines the regularization to be used.
        - gamma: float, regularization coefficient. Used in conjunction with 'penalty'.
        - fit_intercept: bool, whether to add an intercept (bias) term.
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def sigmoid(self, x):
        y=1./(1.+np.exp(-x))
        return y

    def fit(self, X, y, lr=0.00001, tol=1e-7, max_iter=1000):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Initialize coefficients
        self.coef_ = np.ones(X.shape[1])
        # List to store loss values at each iteration
        losses = []

        n_samples = X.shape[0]

        for iteration in range(int(max_iter)):
            linear_output = np.dot(X, self.coef_)
            y_pred = self.sigmoid(linear_output)
            y_pred=np.array(y_pred)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            #y_pred = np.where(y_pred >= 0.5, 1, 0)
            gradient = np.dot(X.T,(y_pred - y)) / n_samples # Calculate the gradient

            if self.penalty == "l1":
                gradient[1:] = gradient[1:]+self.gamma*np.sign(self.coef_[1:])/(2*n_samples)
                loss = loss + np.sum(np.abs(self.coef_[1:])) * self.gamma / (2 * n_samples)
            elif self.penalty == "l2":
                gradient[1:] = gradient[1:]+self.gamma*self.coef_[1:]/n_samples
                loss = loss + np.sum(self.coef_[1:]**2) * self.gamma / (2 * n_samples)
            losses.append(loss)
            norm_diff = np.linalg.norm(gradient)
            if abs(norm_diff).all() < tol:
                break
            self.coef_ = self.coef_ - lr* gradient
        print(losses)
        return losses
    


    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.
        
        Returns:
        - probs: numpy array of shape (n_samples,), prediction probabilities.
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Compute the linear combination of inputs and weights
        linear_output = np.dot(X, self.coef_)
        probs = self.sigmoid(linear_output)  # 使用之前定义的sigmoid函数

        return probs
