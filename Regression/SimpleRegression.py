import numpy as np

from .Scoring import Scoring


class SimpleLinearRegression(Scoring):

    def __init__(self):
        super().__init__()

    def fit(self, X, Y):
        self.m, self.n = X.shape

        ones = np.ones(shape=self.m).reshape(-1, 1)
        X = np.concatenate((ones, X), 1)

        inv_mat = np.linalg.inv(X.transpose().dot(X))

        coefficient = inv_mat.dot(X.transpose()).dot(Y)

        self.intercept_ = coefficient[0]
        self.coef_ = coefficient[1:]

        return self

    def __str__(self):
        res = ""
        res += f"Slope: {list(self.coef_)}, Intercept: {self.intercept_}\n"
        res_p = f"Equation {list(self.coef_)}X + {self.intercept_}"
        if self.intercept_ < 0:
            res_p = f"Equation: Y = {list(self.coef_)}X - {np.abs(self.intercept_)}" # noqa
        res += res_p
        return res
