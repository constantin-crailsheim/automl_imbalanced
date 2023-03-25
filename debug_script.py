
from data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from ImbalancedAutoML import ImbalancedAutoML

dataset_number = 1002
data = Dataset.from_openml(dataset_number)

X = data.features.to_numpy()
y = data.labels.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
automl = ImbalancedAutoML(total_cost=15)

score_vector = []

automl.fit(X_train, y_train, verbose=False, output_name=str(dataset_number))

y_eval = automl.predict(X_test, type="best")

print("Test accuracy of best: {:.3f}".format(balanced_accuracy_score(y_test, y_eval)))

y_eval = automl.predict(X_test, type="ensemble")

print("Test accuracy of ensemble: {:.3f}".format(balanced_accuracy_score(y_test, y_eval)))

y_eval = automl.predict(X_test, type="rf")

print("Test accuracy of RF: {:.3f}".format(balanced_accuracy_score(y_test, y_eval)))

y_eval = automl.predict(X_test, type="gb")

print("Test accuracy of GB: {:.3f}".format(balanced_accuracy_score(y_test, y_eval)))

y_eval = automl.predict(X_test, type="svm")

print("Test accuracy of SVM: {:.3f}".format(balanced_accuracy_score(y_test, y_eval)))
