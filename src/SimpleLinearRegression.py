from typing import Union
from typing_extensions import Self
import numpy as np
import numpy.typing as npt


class SimpleLinearRegression:

    def __init__(self, X: np.ndarray, y: npt.ArrayLike) -> None:
        self.X = X
        self.y = y
        self.intercept_ = None
        self.coef_ = None

    def __str__(self) -> str:
        res = ""
        res += f"Slope: {list(self.coef_)}, Intercept: {self.intercept_}\n"
        res_p = f"Equation {list(self.coef_)}X + {self.intercept_}"
        if self.intercept_ < 0:
            res_p = f"Equation: Y = {list(self.coef_)}X - {np.abs(self.intercept_)}" # noqa
        res += res_p
        return res

    def __repr__(self) -> str:
        return self.__str__()

    def fit(self) -> Self:
        self.m, self.n = self.X.shape
        ones_ = np.ones(shape=self.m).reshape(-1, 1)
        X_cont = np.concatenate((ones_, self.X), 1)
        inv_mat = np.linalg.inv(np.dot(X_cont.T, X_cont))
        mat = np.dot(inv_mat, X_cont.T)
        coefficient = np.dot(mat, self.y)
        self.intercept_ = coefficient[0]
        self.coef_ = coefficient[1:]
        return self

    def predict(self, X: Union[np.ndarray, None] = None) -> npt.ArrayLike:
        if X is None:
            X = self.X
        return X.dot(self.coef_) + self.intercept_

    def score(self, X: Union[np.ndarray, None] = None, y: Union[npt.ArrayLike, None] = None) -> float:
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        y_pred = self.predict(X)
        sum_numerator = np.sum(np.square(y - y_pred))
        sum_denumerator = np.sum(np.square(y - np.mean(y)))
        return (1-(sum_numerator/sum_denumerator))
