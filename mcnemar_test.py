import os
import numpy as np
import json

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn import impute, pipeline, ensemble

from data import Dataset
from configuration import data_ids

from ImbalancedAutoML import ImbalancedAutoML

path = "results/run_1"

random_forest = pipeline.Pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", ensemble.RandomForestClassifier()),
    ]
)

automl = ImbalancedAutoML()

skf = StratifiedKFold(n_splits=3)

def McNemar_test(labels, prediction_1, prediction_2):
    """
    :param labels: the ground truth labels
    :param prediction_1: the prediction results from model 1
    :param prediction_2:  the prediction results from model 2
    :return: the test statistic chi2_Mc
    """
    correct_model1 = labels == prediction_1
    correct_model2 = labels == prediction_2

    A = sum(correct_model1 & correct_model2)
    B = sum(correct_model1 & ~correct_model2)
    C = sum(~correct_model1 & correct_model2)
    D = sum(~correct_model1 & ~correct_model2)

    chi2_Mc = ((abs(B - C) - 1) ** 2) / (B + C)

    return chi2_Mc

for id in data_ids[0:1]:
    dataset = Dataset.from_openml(id)

    X = dataset.features.to_numpy()
    y = dataset.labels.to_numpy()

    mcnemar_test = []

    for train, test in skf.split(X, y):
        X_train, y_train  = X[train], y[train]
        X_test, y_test  = X[test], y[test]

        random_forest.fit(X_train, y_train)

        y_eval1 = random_forest.predict(X_test)

        for cv_fold in range(1,4):

            path_cv = path + '/dataset_{}_cv_{}'.format(id,cv_fold)

            files_in_dir = np.array(os.listdir(path_cv))

            inc_files = np.array(["incumbent" in file_in_dir for file_in_dir in files_in_dir])

            inc_files = files_in_dir[inc_files]

            incumbents_dict = {}

            for file_name in inc_files:
                inc_dict = json.load(open(path_cv + "/" + file_name))["config"]
                if "rf" in file_name:
                    incumbents_dict["rf"] = inc_dict
                elif "gb" in file_name:
                    incumbents_dict["gb"] = inc_dict
                elif "svm" in file_name:
                    incumbents_dict["svm"] = inc_dict
            
            automl.fit_with_hp(X_train, y_train, hp_dict=incumbents_dict, features_dtypes=dataset.features.dtypes)

            y_eval2 = automl.predict(X_test)

            mcnemar_test.append(McNemar_test(y_test, y_eval1, y_eval2))
    
    print("McNemar test for dataset {}: {}".format(id, np.mean(np.array(mcnemar_test))))

   