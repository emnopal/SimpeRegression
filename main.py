from Regression.SimpleRegression import SimpleLinearRegression

import numpy as np
import pandas as pd

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

    model_multi = SimpleLinearRegression()
    model_multi.fit(x_multi, y_multi)
    print(model_multi)
    print(model_multi.predict(x_multi))
    print(model_multi.score(x_multi, y_multi))
    print()

    # Example of linear data

    print("Example of Linear Data")

    x = [3, 21, 22, 34, 54, 34, 55, 67, 89, 99]
    x = np.array(x).reshape(-1, 1)
    y = [1, 10, 14, 34, 44, 48, 57, 67, 79, 90]
    y = np.array(y)

    model = SimpleLinearRegression()
    model.fit(x, y)
    print(model)
    print(model.predict(x))
    print(model.score(x, y))
