# %%

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configuration import data_ids
import json

# %%

baseline_performance_dict = {976: 0.960, 1002: 0.537}
automl_performance_dict = {976: 0.985, 1002: 0.810}

performance_df = pd.DataFrame(columns=["Model"] + list(baseline_performance_dict.keys()))
baseline_performance_dict["Model"] = "Baseline"
automl_performance_dict["Model"] = "AutoML system"

performance_df = performance_df.append(baseline_performance_dict, ignore_index=True)
performance_df = performance_df.append(automl_performance_dict, ignore_index=True)
performance_df.loc[len(performance_df)] = ["Improvement"] + list(performance_df.iloc[1,1:] - performance_df.iloc[0,1:])

print(performance_df.to_latex(index=False))

# baseline_performance_dict = pickle.load(open("results/baseline_performance_dict.pkl", 'rb'))
# automl_performance_dict = pickle.load(open("results/automl_performance_dict.pkl", 'rb'))


# %%

# Insert dataset id here
dataset = 976
total_cost = 3600
outer_cv_folds = 3
x_max = total_cost/outer_cv_folds * 0.4

# Load objects

path_cv1 = "results/dataset_{}_cv_1".format(dataset)
path_cv2 = "results/dataset_{}_cv_2".format(dataset)
path_cv3 = "results/dataset_{}_cv_3".format(dataset)

dehb_objects_cv1 = pickle.load(open(path_cv1 + "/dehb_objects.pkl", 'rb'))
model_cv1 = pickle.load(open(path_cv1 + "/model.pkl", 'rb'))
runtimes_cv1 = pickle.load(open(path_cv1 + "/runtimes.pkl", 'rb'))
trajectories_cv1 = pickle.load(open(path_cv1 + "/trajectories.pkl", 'rb'))

dehb_objects_cv2 = pickle.load(open(path_cv2 + "/dehb_objects.pkl", 'rb'))
model_cv2 = pickle.load(open(path_cv2 + "/model.pkl", 'rb'))
runtimes_cv2 = pickle.load(open(path_cv2 + "/runtimes.pkl", 'rb'))
trajectories_cv2 = pickle.load(open(path_cv2 + "/trajectories.pkl", 'rb'))

dehb_objects_cv3 = pickle.load(open(path_cv3 + "/dehb_objects.pkl", 'rb'))
model_cv3 = pickle.load(open(path_cv3 + "/model.pkl", 'rb'))
runtimes_cv3 = pickle.load(open(path_cv3 + "/runtimes.pkl", 'rb'))
trajectories_cv3 = pickle.load(open(path_cv3 + "/trajectories.pkl", 'rb'))

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

time_rf_cv3 = runtimes_cv3[0].cumsum()
performance_rf_cv3 = -trajectories_cv3[0]
time_gb_cv3 = runtimes_cv3[1].cumsum()
performance_gb_cv3 = -trajectories_cv3[1]
time_svm_cv3 = runtimes_cv3[2].cumsum()
performance_svm_cv3 = -trajectories_cv3[2]

plt.plot(time_rf_cv1, performance_rf_cv1, color="cornflowerblue")
plt.plot(time_rf_cv2, performance_rf_cv2, color="blue")
plt.plot(time_rf_cv3, performance_rf_cv3, color="navy")

plt.plot(time_gb_cv1, performance_gb_cv1, color="orange")
plt.plot(time_gb_cv2, performance_gb_cv2, color="orangered")
plt.plot(time_gb_cv3, performance_gb_cv3, color="firebrick")

plt.plot(time_svm_cv1, performance_svm_cv1, color="lightgreen")
plt.plot(time_svm_cv2, performance_svm_cv2, color="green")
plt.plot(time_svm_cv3, performance_svm_cv3, color="darkgreen")

plt.hlines(y=baseline_performance_dict[dataset], xmin=0, xmax=x_max, colors="dimgrey")

plt.plot(x_max, automl_performance_dict[dataset], marker="o", markersize=5, markeredgecolor="purple", markerfacecolor="purple")

plt.ylim([0.9,1.0])

plt.xlabel("Wallclock time in seconds")
plt.ylabel("Accuracy")
plt.title("Trajectories of dataset with id {}".format(dataset))
plt.legend(["Random forest (CV 1)", "Random forest (CV 2)", "Random forest (CV 3)", "Gradient boosting (CV 1)", "Gradient boosting (CV 2)", "Gradient boosting (CV 3)", "SVM (CV 1)", "SVM (CV 2)", "SVM (CV 3)", "Untuned RF baseline", "AutoML sytem (ECV)"])

if not os.path.exists("trajectory_plots"):
    os.makedirs("trajectory_plots")

# plt.savefig("trajectory_plots/plot_dataset_{}.png".format(dataset))

# %%

inc_hp_rf = pd.DataFrame(columns=["Dataset ID", "CV fold", "imputation_strategy", "sampling_strategy", "scaling_strategy", "criterion", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "class_weight"])
inc_hp_gb = pd.DataFrame(columns=["Dataset ID", "CV fold", "imputation_strategy", "sampling_strategy", "scaling_strategy", "loss", "learning_rate", "criterion", "min_samples_split", "min_samples_leaf", "max_depth"])
inc_hp_svm = pd.DataFrame(columns=["Dataset ID", "CV fold", "imputation_strategy", "sampling_strategy", "scaling_strategy", "C", "kernel", "shrinking", "tol", "class_weight"])

# data_ids
ids = [976]
for id in ids:
    for cv_fold in range(1,4):

        path = 'results/dataset_{}_cv_{}'.format(id,cv_fold)

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

