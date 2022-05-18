import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from src import SimpleLinearRegression


if __name__ == '__main__':

    # Example of multidimensional data
    print("Example of Multidimensional Data")

    datas = pd.DataFrame({
        'one': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'two': [2, 5, 6, 9, 10, 11, 17, 21, 32, 41],
        'three': [3, 5, 12, 17, 21, 48, 57, 69, 72, 81]
    })
    x_multi = datas.drop(['one'], axis=1).values
    y_multi = datas['one'].values

    model_multi = SimpleLinearRegression(x_multi, y_multi)
    model_multi.fit()
    print(model_multi)
    print(model_multi.predict(x_multi))
    print(model_multi.score(x_multi, y_multi))
    print()

    # Comparison to Scikit Linear Regression
    models = LinearRegression()
    models.fit(x_multi, y_multi)
    print(models)
    print(models.predict(x_multi))
    print(models.score(x_multi, y_multi))
    print(models.intercept_)
    print(models.coef_)
    print()

    # Example of linear data
    print("Example of Linear Data")

    x = [3, 21, 22, 34, 54, 34, 55, 67, 89, 99]
    x = np.array(x).reshape(-1, 1)
    y = [1, 10, 14, 34, 44, 48, 57, 67, 79, 90]
    y = np.array(y)

    model = SimpleLinearRegression(x, y)
    model.fit()
    print(model)
    print(model.predict(x))
    print(model.score(x, y))
    print()

    # Comparison to Scikit Linear Regression
    models_ = LinearRegression()
    models_.fit(x, y)
    print(models_)
    print(models_.predict(x))
    print(models_.score(x, y))
    print(models_.intercept_)
    print(models_.coef_)
    print()
