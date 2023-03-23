import numpy as np
import time

from sklearn import dummy, impute, pipeline, tree, ensemble
from sklearn.model_selection import KFold, cross_val_score

from configuration import data_ids, scoring
from data import Dataset
from ImbalancedAutoML import ImbalancedAutoML

# This file runs a small benchmark on the imbalanced datasets
# comparing a classification tree and a majority vote baseline

featurless = dummy.DummyClassifier()

random_forest = pipeline.Pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", ensemble.RandomForestClassifier()),
    ]
)

total_cost = 300
cv_n_splits = 2

cv = KFold(n_splits=cv_n_splits, shuffle=True, random_state=42)

automl = ImbalancedAutoML(total_cost=total_cost/cv_n_splits)

for id in data_ids[7:]:
    dataset = Dataset.from_openml(id)

    print(f"Running Classification tree on {dataset.name}")

    X = dataset.features.to_numpy()
    y = dataset.labels.to_numpy()

    scores_random_forest = cross_val_score(random_forest, X, y, scoring=scoring, cv=cv)
    print("Balanced Accuracy of classification random forest: {:.3f}".format(np.mean(scores_random_forest)))

    start = time.time()
    scores_automl = cross_val_score(automl, X, y, scoring=scoring, cv=cv, fit_params={"output_name": str(id)})
    print("Balanced Accuracy of AutoML system: {:.3f}\n".format(np.mean(scores_automl)))
    print("Total time taken in seconds: {:.1f}".format(time.time()-start))
    
