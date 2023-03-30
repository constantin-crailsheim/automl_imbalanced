import os
import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn import impute, pipeline, ensemble

from data import Dataset
from configuration import data_ids, seed, outer_cv_folds
from utils import McNemar_test
from ImbalancedAutoML import ImbalancedAutoML

path = "results/run_2"

random_forest = pipeline.Pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", ensemble.RandomForestClassifier(random_state=seed)),
    ]
)

automl = ImbalancedAutoML()

skf = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=seed)

for id in data_ids:
    dataset = Dataset.from_openml(id)

    X = dataset.features.to_numpy()
    y = dataset.labels.to_numpy()
 
    scores_baseline = []
    scores_automl = []

    y_test_all = []
    y_eval_rf_all = []
    y_eval_automl_all = []

    cv_fold = 1

    for train, test in skf.split(X, y):
        X_train, y_train  = X[train], y[train]
        X_test, y_test  = X[test], y[test]

        random_forest.fit(X_train, y_train)

        y_eval_rf = random_forest.predict(X_test)
        scores_baseline.append(balanced_accuracy_score(y_test, y_eval_rf))

        path_cv = path + '/dataset_{}_cv_{}'.format(id,cv_fold)

        automl = pickle.load(open(path_cv + "/voting_classifier.pkl", 'rb'))

        y_eval_automl = automl.predict(X_test)
        scores_automl.append(balanced_accuracy_score(y_test, y_eval_automl))

        y_test_all.append(y_test)
        y_eval_rf_all.append(y_eval_rf)
        y_eval_automl_all.append(y_eval_automl)

    y_test_all = np.concatenate(y_test_all)
    y_eval_rf_all = np.concatenate(y_eval_rf_all)
    y_eval_automl_all = np.concatenate(y_eval_automl_all)

    print("CV balanced accuracy of classification random forest: {:.3f}".format(np.mean(np.array(scores_baseline))))
    print("CV balanced accuracy of AutoML system: {:.3f}".format(np.mean(np.array(scores_automl))))
    print("McNemar test for dataset {}: {}\n".format(id, McNemar_test(y_test_all, y_eval_rf_all, y_eval_automl_all)))

   