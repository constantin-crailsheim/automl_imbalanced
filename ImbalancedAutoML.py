from configuration import scoring

from sklearn.base import ClassifierMixin

from sklearn.model_selection import KFold, cross_val_score

from sklearn import impute, pipeline, tree, ensemble
from sklearn.metrics import balanced_accuracy_score

from imblearn import under_sampling, over_sampling


class ImbalancedAutoML(ClassifierMixin):
    def __init__(self) -> None:
        super().__init__()

        self.hyperparameter = None

        self.sampling_strategies = [
            over_sampling.SMOTE(random_state=42),
            under_sampling.TomekLinks(),
            under_sampling.CondensedNearestNeighbour()]

        self.model = pipeline.Pipeline(
            steps=[
                ("imputer", impute.SimpleImputer()),
                ("estimator", ensemble.RandomForestClassifier()),
            ]
        )

        self.cv = KFold(n_splits=2, shuffle=True, random_state=42)


    def fit(self, X=None, y=None, verbose=False, path_results=None, time_limit=3600):

        for sampling_strategy in self.sampling_strategies:
            X, y = self.smote.fit_resample(X, y)
            
            # How to train model and pick correct model?

            score = cross_val_score(self.model, X, y, scoring=scoring, cv=self.cv)

            # self.model.fit(X, y)

        return self

    def predict(self, X=None):

        return self.model.predict(X)
