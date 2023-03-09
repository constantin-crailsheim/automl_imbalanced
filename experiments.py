# %%

import numpy as np
import pandas as pd

from data import Dataset
from configuration import data_ids
from sklearn import impute, pipeline, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# %%

data = Dataset.from_openml(976)

X = data.features.to_numpy()
y = data.labels.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %%

from ImbalancedAutoML import ImbalancedAutoML

automl = ImbalancedAutoML()

score_vector = []

for _ in range(1):
    automl.fit(X_train, y_train)

    y_eval = automl.predict(X_test)

    score_vector.append(balanced_accuracy_score(y_test, y_eval))

print(np.array(score_vector).mean())

# %%

for data_id in data_ids:
    data = Dataset.from_openml(data_id)

    X = data.features.to_numpy()
    y = data.labels.to_numpy()

    df = pd.concat([pd.DataFrame(y, columns=["Label"]), pd.DataFrame(X)], axis=1)

    print(df.loc[:,"Label"].value_counts()/len(df))
    

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
