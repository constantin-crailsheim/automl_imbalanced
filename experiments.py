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

dataset_number = 976
data = Dataset.from_openml(dataset_number)

X = data.features.to_numpy()
y = data.labels.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %%

automl = ImbalancedAutoML(total_cost=15)

score_vector = []

automl.fit(X_train, y_train, number_restarts=3, verbose=False, output_name=str(dataset_number))

y_eval = automl.predict(X_test)

print("Test accuracy: {:.3f}".format(balanced_accuracy_score(y_test, y_eval)))

dehb = automl.get_dehb()

# %%

for i in range(3):
    print("Incumbent configuration with train accuracy of {:.3f}:".format(-dehb[i].get_incumbents()[1]))
    print(dehb[i].vector_to_configspace(dehb[i].inc_config).get_dictionary())

# %%

restart_number = 2

histories = automl.get_histories()

last_eval = histories[restart_number][-1]
config, score, cost, budget, _info = last_eval

dehb[restart_number].vector_to_configspace(config)

# %%

automl = ImbalancedAutoML(total_cost=15)

cv = KFold(n_splits=2, shuffle=True, random_state=42)

scores_dummy = cross_val_score(automl, X, y, scoring=scoring, cv=cv)
print(f"Balanced Accuracy of featurless baseline: {np.mean(scores_dummy)}")
    

# %%

classtree = pipeline.Pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", tree.DecisionTreeClassifier()),
    ]
)

score_vector = []

for _ in range(100):

    classtree.fit(X_train, y_train)

    y_eval = classtree.predict(X_test)

    score_vector.append(balanced_accuracy_score(y_test, y_eval))

print(np.array(score_vector).mean())


# %%

from imblearn import under_sampling, over_sampling

sm = over_sampling.SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(X_train, y_train)

classtree.fit(X_res, y_res)

y_eval = classtree.predict(X_test)

print(balanced_accuracy_score(y_test, y_eval))

# %%
# %%
