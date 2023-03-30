# Import packages
import time
import pickle
import numpy as np

from sklearn import impute, pipeline, ensemble
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

# Import objects and classes from repo
from configuration import data_ids, scoring, output_path, total_cost, outer_cv_folds, seed
from data import Dataset
from utils import McNemar_test, delete_large_file
from ImbalancedAutoML import ImbalancedAutoML

# This file runs a benchmark of the RandomForestClassifier baseline
# compared to the AutoML system

random_forest = pipeline.Pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", ensemble.RandomForestClassifier(random_state=seed)),
    ]
)

# Set global random seed
np.random.seed(seed)

# Initialise 3-fold cross-validation which ensures that share of targets is preserved in each fold.
scv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=seed)

# Initialise ImbalancedAutoML object with maximum cost of 1200 seconds
automl = ImbalancedAutoML(total_cost=total_cost/outer_cv_folds, random_state=seed)

# Initialise dicts to store externally cross-validated performance
baseline_performance_dict = {}
automl_performance_dict = {}
mcnemar_dict = {}

for id in data_ids:

    dataset = Dataset.from_openml(id)

    print(f"Comparing random forest baseline with AutoML system on {dataset.name} with ID {id}")

    X = dataset.features.to_numpy()
    y = dataset.labels.to_numpy()

    scores_baseline = []
    scores_automl = []
    time_auto_ml = []
    mcnemar = []

    y_test_all = []
    y_eval_rf_all = []
    y_eval_automl_all = []

    cv_fold = 1
    for train, test in scv.split(X, y):
        X_train, y_train  = X[train], y[train]
        X_test, y_test  = X[test], y[test]

        # Check performance of baseline on cross-validation fold
        random_forest.fit(X_train, y_train)
        y_eval_rf = random_forest.predict(X_test)
        scores_baseline.append(balanced_accuracy_score(y_test, y_eval_rf))

        # Check performance of AutoML system with cross-validation and track time taken
        start = time.time()
        automl.fit(X_train, y_train, output_path = output_path, output_name = str(id), features_dtypes = dataset.features.dtypes, cv_fold=cv_fold)
        y_eval_automl = automl.predict(X_test)
        scores_automl.append(balanced_accuracy_score(y_test, y_eval_automl))
        time_auto_ml.append(time.time() - start)

        mcnemar.append(McNemar_test(y_test, y_eval_rf, y_eval_automl))
        
        y_test_all.append(y_test)
        y_eval_rf_all.append(y_eval_rf)
        y_eval_automl_all.append(y_eval_automl)
        
        cv_fold += 1

    y_test_all = np.concatenate(y_test_all)
    y_eval_rf_all = np.concatenate(y_eval_rf_all)
    y_eval_automl_all = np.concatenate(y_eval_automl_all)

    mcnemar.append(McNemar_test(y_test_all, y_eval_rf_all, y_eval_automl_all))

    baseline_performance_dict[id] = scores_baseline + [np.mean(np.array(scores_baseline))]
    print("CV balanced accuracy of classification random forest: {:.3f}".format(np.mean(np.array(scores_baseline))))

    automl_performance_dict[id] = scores_automl + [np.mean(np.array(scores_automl))]
    print("CV balanced accuracy of AutoML system: {:.3f}".format(np.mean(np.array(scores_automl))))
    print("Total time taken for AutoML system: {:.1f}\n".format(np.sum(np.array(time_auto_ml))))

    # Check  McNemar test
    mcnemar_dict[id] = mcnemar
    print("Test statistic of McNemar test: {:.2f}".format(mcnemar_dict[id][3]))

# Store dicts of externally cross-validated performance
pickle.dump(baseline_performance_dict, open(output_path + "/baseline_performance_dict.pkl", 'wb'))
pickle.dump(automl_performance_dict, open(output_path + "/automl_performance_dict.pkl", 'wb'))
pickle.dump(mcnemar_dict, open(output_path + "/mcnemar_dict.pkl", 'wb'))

# Delete large files not needed for further analysis
delete_large_file(output_path)