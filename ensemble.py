# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, clone, ClassifierMixin
from sklearn.svm import OneClassSVM
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class Bagging:

    def __init__(
        self, estimator: BaseEstimator, n_estimators: int = 10, percentage: float = 0.8
    ):
        """
        Initialize the Bagging class.

        :param estimator: The base estimator to be used.
        :param n_estimators: The number of base estimators.
        :param percentage: The percentage of data to be used for training each base estimator.
        """
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.percentage = percentage

        self.estimators = [self._clone_estimator() for _ in range(n_estimators)]

    def _clone_estimator(self) -> BaseEstimator:
        """
        Clone the base estimator.

        :return: A new instance of the base estimator.
        """
        return clone(self.estimator)

    def get_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate feature samples for each estimator.

        :param X: The input data.
        :return: A list of feature samples.
        """
        features_samples = []
        num_samples = int(self.percentage * len(X))

        for _ in range(self.n_estimators):
            feat = []
            sample_indices = np.random.choice(len(X), num_samples, replace=False)
            for idx in sample_indices:
                feat.append(X[idx])

            features_samples.append(feat)

        return np.array(features_samples)

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the ensemble of estimators.

        :param X: The input data.
        """
        features = self.get_features(X)
        for i in range(self.n_estimators):
            print(f"Training estimator {i+1}")
            self.estimators[i].fit(features[i])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the ensemble of estimators.

        :param X: The input data.
        :return: The predicted labels.
        """
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        return self.vote(predictions)

    def vote(self, predictions: np.ndarray) -> np.ndarray:
        """
        Aggregate predictions using majority voting.

        :param predictions: The predictions from each estimator.
        :return: The aggregated predictions.
        """
        summed_predictions = np.sum(predictions, axis=0)

        return np.where(summed_predictions >= 0, 1, -1)


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


if __name__ == "__main__":
    # bag = Bagging(estimator=OneClassSVM())
    X = np.random.randn(10, 5)
    # feat_samples = bag.get_features(X)
    # print(feat_samples)

    # bag.fit(X)
    # pred = bag.predict(X)
    # print(pred)
