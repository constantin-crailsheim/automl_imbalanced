# %%

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configuration import data_ids
import json

# %%

# Insert dataset id here
dataset = 976

baseline_performance_dict = {976: 0.960, 1002: 0.537}
automl_performance_dict = {976: 0.985, 1002: 0.810}

# baseline_performance_dict = pickle.load(open("results/baseline_performance_dict.pkl", 'rb'))
# automl_performance_dict = pickle.load(open("results/automl_performance_dict.pkl", 'rb'))

# Load objects

path_cv1 = "results/dataset_{}_cv1".format(dataset)
path_cv2 = "results/dataset_{}_cv2".format(dataset)

dehb_objects_cv1 = pickle.load(open(path_cv1 + "/dehb_objects.pkl", 'rb'))
model_cv1 = pickle.load(open(path_cv1 + "/model.pkl", 'rb'))
runtimes_cv1 = pickle.load(open(path_cv1 + "/runtimes.pkl", 'rb'))
trajectories_cv1 = pickle.load(open(path_cv1 + "/trajectories.pkl", 'rb'))

dehb_objects_cv2 = pickle.load(open(path_cv2 + "/dehb_objects.pkl", 'rb'))
model_cv2 = pickle.load(open(path_cv2 + "/model.pkl", 'rb'))
runtimes_cv2 = pickle.load(open(path_cv2 + "/runtimes.pkl", 'rb'))
trajectories_cv2 = pickle.load(open(path_cv2 + "/trajectories.pkl", 'rb'))

# Plot performance over time

time_rf_cv1 = runtimes_cv1[0].cumsum()
performance_rf_cv1 = -trajectories_cv1[0]
time_gb_cv1 = runtimes_cv1[1].cumsum()
performance_gb_cv1 = -trajectories_cv1[1]
time_svm_cv1 = runtimes_cv1[2].cumsum()
performance_svm_cv1 = -trajectories_cv1[2]

time_rf_cv2 = runtimes_cv2[0].cumsum()
performance_rf_cv2 = -trajectories_cv2[0]
time_gb_cv2 = runtimes_cv2[1].cumsum()
performance_gb_cv2 = -trajectories_cv2[1]
time_svm_cv2 = runtimes_cv2[2].cumsum()
performance_svm_cv2 = -trajectories_cv2[2]

plt.plot(time_rf_cv1, performance_rf_cv1, color="cornflowerblue")
plt.plot(time_rf_cv2, performance_rf_cv2, color="blue")
# color=navy

plt.plot(time_gb_cv1, performance_gb_cv1, color="orange")
plt.plot(time_gb_cv2, performance_gb_cv2, color="orangered")
# color=firebrick

plt.plot(time_svm_cv1, performance_svm_cv1, color="lightgreen")
plt.plot(time_svm_cv2, performance_svm_cv2, color="green")
# color=darkgreen

plt.hlines(y=baseline_performance_dict[dataset], xmin=0, xmax=600, colors="dimgrey")

plt.plot(600, automl_performance_dict[dataset], marker="o", markersize=5, markeredgecolor="purple", markerfacecolor="purple")

plt.xlabel("Wallclock time in seconds")
plt.ylabel("Accuracy")
plt.title("Trajectories of dataset with id {}".format(dataset))
plt.legend(["Random forest (CV 1)", "Random forest (CV 2)", "Gradient boosting (CV 1)", "Gradient boosting (CV 2)", "SVM (CV 1)", "SVM (CV 2)", "Untuned RF baseline", "AutoML sytem (ECV)"])

if not os.path.exists("trajectory_plots"):
    os.makedirs("trajectory_plots")

plt.savefig("trajectory_plots/plot_dataset_{}.png".format(dataset))

# %%

inc_hp_rf = pd.DataFrame(columns=["Dataset ID", "CV fold", "criterion", "imputation_strategy", "max_depth", "max_features", "min_samples_leaf", "min_samples_split", "n_estimators", "sampling_strategy"])
inc_hp_gb = pd.DataFrame(columns=["Dataset ID", "CV fold", "criterion", "imputation_strategy", "learning_rate", "loss", "min_samples_leaf", "min_samples_split", "n_estimators", "sampling_strategy"])
inc_hp_svm = pd.DataFrame(columns=["Dataset ID", "CV fold", "C", "class_weight", "imputation_strategy", "kernel", "sampling_strategy", "shrinking"])

# data_ids
ids = [976, 1002]
for id in ids:
    for cv_fold in range(1,3):

        path = 'results/dataset_{}_cv{}'.format(id,cv_fold)

        files_in_dir = np.array(os.listdir(path))

        inc_files = np.array(["incumbent" in file_in_dir for file_in_dir in files_in_dir])

        inc_files = files_in_dir[inc_files]

        for file_name in inc_files:
            hp_dict = json.load(open(path + "/" + file_name))["config"]
            hp_dict["Dataset ID"] = id
            hp_dict["CV fold"] = cv_fold

            if "rf" in file_name:
                inc_hp_rf = inc_hp_rf.append(hp_dict, ignore_index=True)
            elif "gb" in file_name:
                inc_hp_gb = inc_hp_gb.append(hp_dict, ignore_index=True)
            elif "svm" in file_name:
                inc_hp_svm = inc_hp_svm.append(hp_dict, ignore_index=True)

# %%

display(inc_hp_rf)

# %%

display(inc_hp_gb)

# %%

display(inc_hp_svm)

# %%
