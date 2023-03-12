import numpy as np
from configuration import data_ids, scoring
from data import Dataset
from sklearn import dummy, impute, pipeline, tree
from sklearn.model_selection import KFold, cross_val_score

from ImbalancedAutoML import ImbalancedAutoML

# This file runs a small benchmark on the imbalanced datasets
# comparing a classification tree and a majority vote baseline

featurless = dummy.DummyClassifier()

tree = pipeline.Pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", tree.DecisionTreeClassifier()),
    ]
)

cv_n_splits = 2

cv = KFold(n_splits=cv_n_splits, shuffle=True, random_state=42)

automl = ImbalancedAutoML(total_cost=600/cv_n_splits)

for id in data_ids[:3]:
    dataset = Dataset.from_openml(id)

    print(f"Running Classification tree on {dataset.name}")

    X = dataset.features.to_numpy()
    y = dataset.labels.to_numpy()

    scores_dummy = cross_val_score(featurless, X, y, scoring=scoring, cv=cv)
    print("Balanced Accuracy of featurless baseline: {:.3f}".format(np.mean(scores_dummy)))

    scores_tree = cross_val_score(tree, X, y, scoring=scoring, cv=cv)
    print("Balanced Accuracy of classification tree: {:.3f}".format(np.mean(scores_tree)))

    scores_automl = cross_val_score(automl, X, y, scoring=scoring, cv=cv, fit_params={"number_restarts": 3, "output_name": str(id)})
    print("Balanced Accuracy of AutoML system: {:.3f}\n".format(np.mean(scores_automl)))
    
