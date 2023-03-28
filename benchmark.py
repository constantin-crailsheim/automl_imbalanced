# Import other modules
import numpy as np
import time
import pickle
import os

from sklearn import impute, pipeline, ensemble
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Import objects and classes from repo
from configuration import data_ids, scoring
from data import Dataset
from ImbalancedAutoML import ImbalancedAutoML

# This file runs a benchmark of the RandomForestClassifier baseline
# compared to the AutoML system

random_forest = pipeline.Pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", ensemble.RandomForestClassifier()),
    ]
)

# Set total maximum cost of optimisation in seconds.
total_cost = 30
# Set number of CV splits.
cv_n_splits = 3

# Initialise 3-fold cross-validation which ensures that share of targets is preserved in each fold.
cv = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)

# Initialise ImbalancedAutoML object with maximum cost of 1200 seconds
automl = ImbalancedAutoML(total_cost=total_cost/cv_n_splits)

# Initialise dicts to store externally cross-validated performance
baseline_performance_dict = {}
automl_performance_dict = {}

for id in data_ids[0:1]:
    dataset = Dataset.from_openml(id)

    print(f"Comparing random forest baseline with AutoML system on {dataset.name} with ID {id}")

    X = dataset.features.to_numpy()
    y = dataset.labels.to_numpy()

    # Check performance of baseline with cross-validation
    scores_random_forest = cross_val_score(random_forest, X, y, scoring=scoring, cv=cv)
    baseline_performance_dict[id] = np.mean(scores_random_forest)
    print("Balanced Accuracy of classification random forest: {:.3f}".format(np.mean(scores_random_forest)))

    # Check performance of AutoML system with cross-validation and track time taken
    start = time.time()
    scores_automl = cross_val_score(automl, X, y, scoring=scoring, cv=cv, fit_params={"output_name": str(id), "features_dtypes": dataset.features.dtypes})
    automl_performance_dict[id] = np.mean(scores_automl)
    print("Balanced Accuracy of AutoML system: {:.3f}".format(np.mean(scores_automl)))
    print("Total time taken in seconds: {:.1f}\n".format(time.time()-start))

# Store dicts of externally cross-validated performance
pickle.dump(baseline_performance_dict, open("results/run_x/baseline_performance_dict.pkl", 'wb'))
pickle.dump(automl_performance_dict, open("results/run_x/automl_performance_dict.pkl", 'wb'))

# Renames the results folder after successful benchmark run such that information is not overwritten in the next run 
for i in range(1,1000):
    results_path = "results/run_{}".format(i)
    if not os.path.exists(results_path):
        os.rename("results/run_x", results_path)
        break
