import numpy as np

from .Prediction import Prediction


class Scoring(Prediction):

    def __init__(self):
        super().__init__()

    def prediction_score(self, Y_pred, Y_actual):
        u = np.sum(np.square(Y_actual - Y_pred))
        v = np.sum(np.square(Y_actual - np.mean(Y_actual)))
        return (1-(u/v))

    def regression_score(self, X, Y):
        Y_pred = self.predict(X)
        u = np.sum(np.square(Y - Y_pred))
        v = np.sum(np.square(Y - np.mean(Y)))
        return (1-(u/v))

    def score(self, X, Y):
        return self.regression_score(X, Y)

    def r2_score(self, X, Y):
        return self.regression_score(X, Y)

    def r_score(self, X, Y):
        return np.sqrt(self.regression_score(X, Y))
