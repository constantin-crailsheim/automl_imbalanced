import numpy as np
import time
import pickle

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

total_cost = 600
cv_n_splits = 3

cv = KFold(n_splits=cv_n_splits, shuffle=True, random_state=42)

automl = ImbalancedAutoML(total_cost=total_cost/cv_n_splits)

baseline_performance_dict = {}
automl_performance_dict = {}

for id in data_ids[0:1]:
    dataset = Dataset.from_openml(id)

    print(f"Running Classification tree on {dataset.name}")

    X = dataset.features.to_numpy()
    y = dataset.labels.to_numpy()

    scores_random_forest = cross_val_score(random_forest, X, y, scoring=scoring, cv=cv)
    baseline_performance_dict[id] = np.mean(scores_random_forest)
    print("Balanced Accuracy of classification random forest: {:.3f}".format(np.mean(scores_random_forest)))

    start = time.time()
    scores_automl = cross_val_score(automl, X, y, scoring=scoring, cv=cv, fit_params={"output_name": str(id)})
    automl_performance_dict[id] = np.mean(scores_automl)
    print("Balanced Accuracy of AutoML system: {:.3f}\n".format(np.mean(scores_automl)))
    print("Total time taken in seconds: {:.1f}".format(time.time()-start))


pickle.dump(baseline_performance_dict, open("results/baseline_performance_dict.pkl", 'wb'))
pickle.dump(automl_performance_dict, open("results/automl_performance_dict.pkl", 'wb'))
