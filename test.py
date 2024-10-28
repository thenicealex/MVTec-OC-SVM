import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class OneClassSVMBoost(BaseEstimator, ClassifierMixin):
    """
    A Boosting implementation using OneClassSVM as the base estimator.

    Parameters:
    -----------
    n_estimators : int, default=10
        The number of boosting stages to perform
    learning_rate : float, default=1.0
        Learning rate shrinks the contribution of each classifier
    base_estimator_params : dict, default=None
        Parameters to initialize OneClassSVM
    """

    def __init__(self, n_estimators=10, learning_rate=1.0, base_estimator_params=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_estimator_params = base_estimator_params or {}

    def fit(self, X, y):
        """
        Fit the boosting model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (1 for inliers, -1 for outliers)

        Returns:
        --------
        self : object
            Returns self.
        """
        # Check inputs
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        # Initialize arrays to store boosting information
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples

        # Boosting iterations
        for iboost in range(self.n_estimators):
            # Create and train base estimator
            estimator = OneClassSVM(**self.base_estimator_params)

            # Sample indices based on weights
            indices = np.random.choice(
                n_samples, size=n_samples, replace=True, p=sample_weights
            )
            sample_X = X[indices]
            sample_y = y[indices]

            # Fit estimator
            estimator.fit(sample_X)

            # Predict on full dataset
            predictions = estimator.predict(X)

            # Calculate error
            incorrect = predictions != y
            error = np.sum(incorrect * sample_weights)

            # Calculate estimator weight
            estimator_weight = self.learning_rate * np.log(
                (1 - error) / max(error, 1e-10)
            )
            self.estimator_weights_[iboost] = estimator_weight

            # Update sample weights
            sample_weights *= np.exp(estimator_weight * incorrect)
            sample_weights /= np.sum(sample_weights)

            # Store the estimator
            self.estimators_.append(estimator)

        return self

    def predict(self, X):
        """
        Predict using the boosted model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        """
        check_is_fitted(self)
        X = check_array(X)

        # Initialize predictions
        scores = np.zeros(X.shape[0])

        # Aggregate predictions from all estimators
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            scores += weight * estimator.predict(X)

        return np.sign(scores)

    def decision_function(self, X):
        """
        Compute the decision function for the samples.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to compute decision function for

        Returns:
        --------
        scores : array-like of shape (n_samples,)
            Decision function values
        """
        check_is_fitted(self)
        X = check_array(X)

        scores = np.zeros(X.shape[0])
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            scores += weight * estimator.decision_function(X)

        return scores


# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )
    y = 2 * (y - 0.5)  # Convert to {-1, 1}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the boosted model
    model = OneClassSVMBoost(
        n_estimators=10,
        learning_rate=0.1,
        base_estimator_params={"kernel": "rbf", "nu": 0.1},
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
