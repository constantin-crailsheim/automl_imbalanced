# %%

import numpy as np
import pandas as pd

from data import Dataset
from configuration import data_ids, scoring
from sklearn import impute, pipeline, tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score

from ImbalancedAutoML import ImbalancedAutoML

# %%

for id in data_ids:
    data = Dataset.from_openml(id)

    X = data.features.to_numpy()
    # y = data.labels.to_numpy()

    print(X.shape)
    # cat_feat1 = 0
    # cat_feat2 = 0

    # for col in range(X.shape[1]):
    #     if len(np.unique(X[:,col])) < 30:
    #         cat_feat1 += 1
    #     if np.all(np.logical_or(X[:,col] % 1 == 0, np.isnan(X[:,col] % 1))):
    #         cat_feat2 += 1
    
    # print("For dataset {} with {} observations, there are ({}, {}) categorical features out of {} features.".format(id, X.shape[0], cat_feat1, cat_feat2, X.shape[1]))


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %%

automl = ImbalancedAutoML(total_cost=5)

score_vector = []

automl.fit(X_train, y_train, number_restarts=1, verbose=False, output_name=str(dataset_number))

y_eval = automl.predict(X_test, type="best")

print("Test accuracy: {:.3f}".format(balanced_accuracy_score(y_test, y_eval)))

y_eval = automl.predict(X_test, type="ensemble")

print("Test accuracy: {:.3f}".format(balanced_accuracy_score(y_test, y_eval)))

# %%

y_eval = automl.predict(X_test, type=3)

print("Test accuracy: {:.3f}".format(balanced_accuracy_score(y_test, y_eval)))

# %%

dehb = automl.get_dehb()

for i in range(3):
    print("Incumbent configuration with train accuracy of {:.3f}:".format(-dehb[i].get_incumbents()[1]))
    print(dehb[i].vector_to_configspace(dehb[i].inc_config).get_dictionary())


# %%

# Checks on imbalanced data:

from imblearn import under_sampling, over_sampling, combine

id = 976

data = Dataset.from_openml(id)

X = data.features.to_numpy()
y = data.labels.to_numpy()

cat_feat = []

for col in range(X.shape[1]):
    if np.all(np.logical_or(X[:,col] % 1 == 0, np.isnan(X[:,col] % 1))):
        cat_feat += [col]

sm = over_sampling.SMOTE(random_state=42)

smnc = over_sampling.SMOTENC(random_state=42, categorical_features=cat_feat)

smt = combine.SMOTETomek(random_state=42)

X_res, y_res = sm.fit_resample(X, y)

X_res_nc, y_res_nc = smnc.fit_resample(X, y)

X_res_t, y_res_t = smt.fit_resample(X, y)

X_res[:,cat_feat]= np.round(X_res[:,cat_feat])

print(np.mean(y == "N"))
# print(np.mean(y_res == "N"))

# %%

import numpy as np
from imblearn import over_sampling
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

id = 976

data = Dataset.from_openml(id)

X = data.features.to_numpy()
y = data.labels.to_numpy()

cat_feat = (data.features.dtypes == int).values

sm = over_sampling.SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(X, y)

ct = ColumnTransformer([("round", FunctionTransformer(np.round), cat_feat),
                        ("identity", FunctionTransformer(), )])

X_res_cat = ct.fit_transform(X_res)

print("a")

# %%

import pickle 

history = pickle.load(open("results/dataset_976_cv_1/history_976_model_rf_2023_03_26-05_43_44_PM.pkl", 'rb'))

# %%

for i in range(len(history)):
    print(history[i][3])

# %%

for id in [1019]:
    data = Dataset.from_openml(id)
    X = data.features.to_numpy()
    print(np.mean(X, axis=0))
    print(np.var(X, axis=0))


# %%
