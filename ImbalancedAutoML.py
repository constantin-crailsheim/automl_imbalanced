from sklearn.base import ClassifierMixin


class ImbalancedAutoML(ClassifierMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X=None, y=None):
        
        raise (NotImplementedError)

    def predict(self, X=None):
        raise (NotImplementedError)
