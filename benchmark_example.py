import numpy as np
from configuration import data_ids, scoring
from data import Dataset
from sklearn import dummy, impute, pipeline, tree
from sklearn.model_selection import KFold, cross_val_score

# This file runs a small benchmark on the imbalanced datasets
# comparing a classification tree and a majority vote baseline

tree = pipeline.Pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", tree.DecisionTreeClassifier()),
    ]
)

featurless = dummy.DummyClassifier()

for id in data_ids:
    dataset = Dataset.from_openml(id)

    print(f"Running Classification tree on {dataset.name}")

    X = dataset.features.to_numpy()
    y = dataset.labels.to_numpy()

    cv = KFold(n_splits=2, shuffle=True, random_state=42)

    scores_dummy = cross_val_score(featurless, X, y, scoring=scoring, cv=cv)
    print(f"Balanced Accuracy of featurless baseline: {np.mean(scores_dummy)}")

    scores_tree = cross_val_score(tree, X, y, scoring=scoring, cv=cv)
    print(f"Balanced Accuracy of classification tree: {np.mean(scores_tree)}\n")
