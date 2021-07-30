class Prediction(object):

    def __init__(self):
        pass

    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_
