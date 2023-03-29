
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configuration import data_ids, output_path, total_cost, outer_cv_folds, y_min_dict, y_max_dict
import json

# Store Latex table of benchmark perfomance across all datasets

baseline_performance_dict = pickle.load(open(output_path + "/baseline_performance_dict.pkl", 'rb'))
automl_performance_dict = pickle.load(open(output_path + "/automl_performance_dict.pkl", 'rb'))
# mcnemar_dict = pickle.load(open(output_path + "/mcnemar_dict.pkl", 'rb'))

performance_df = pd.DataFrame(columns=["Model"] + list(baseline_performance_dict.keys()))
baseline_performance_dict["Model"] = "Baseline"
automl_performance_dict["Model"] = "AutoML system"
# mcnemar_dict["Model"] = "McNemar test"

# for id in data_ids[:1]:
#     baseline_performance_dict[id] = baseline_performance_dict[id][3]
#     automl_performance_dict[id] = automl_performance_dict[id][3]
#     mcnemar_dict[id] = mcnemar_dict[id][3]

performance_df = pd.concat([performance_df, pd.DataFrame([baseline_performance_dict])], ignore_index=True)
performance_df = pd.concat([performance_df, pd.DataFrame([automl_performance_dict])], ignore_index=True)
performance_df.loc[len(performance_df)] = ["Improvement"] + list(performance_df.iloc[1,1:].to_numpy() - performance_df.iloc[0,1:])
# performance_df = pd.concat([performance_df, pd.DataFrame([mcnemar_dict])], ignore_index=True)

performance_df.style.hide(axis="index").format(precision=3).to_latex(output_path + "/performance_table.txt") # float_format="{:0.3f}".format

# Store plots of trajectories over runtime for each dataset

x_max = total_cost/outer_cv_folds * 0.4

# Load objects

for dataset in data_ids:

    path_cv1 = output_path + "/dataset_{}_cv_1".format(dataset)
    path_cv2 = output_path + "/dataset_{}_cv_2".format(dataset)
    path_cv3 = output_path + "/dataset_{}_cv_3".format(dataset)

    runtimes_cv1 = pickle.load(open(path_cv1 + "/runtimes.pkl", 'rb'))
    trajectories_cv1 = pickle.load(open(path_cv1 + "/trajectories.pkl", 'rb'))

    runtimes_cv2 = pickle.load(open(path_cv2 + "/runtimes.pkl", 'rb'))
    trajectories_cv2 = pickle.load(open(path_cv2 + "/trajectories.pkl", 'rb'))

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

    fig, axs = plt.subplots(1,1)

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

    plt.ylim([y_min_dict[dataset],y_max_dict[dataset]])

    plt.xlabel("Wallclock time in seconds")
    plt.ylabel("Balanced accuracy")
    plt.title("Trajectories of dataset {}".format(dataset))
    plt.legend(["Random forest (CV 1)", "Random forest (CV 2)", "Random forest (CV 3)", "Gradient boosting (CV 1)", "Gradient boosting (CV 2)", "Gradient boosting (CV 3)", "SVM (CV 1)", "SVM (CV 2)", "SVM (CV 3)", "RF baseline (ECV)", "AutoML sytem (ECV)"])

    if not os.path.exists(output_path + "/trajectory_plots"):
        os.makedirs(output_path + "/trajectory_plots")

    plt.savefig(output_path + "/trajectory_plots/plot_dataset_{}.png".format(dataset))


# Store Latex table of incumbent hyperparameter for each algorithm for all datasets

inc_hp_rf = pd.DataFrame(columns=["Dataset ID", "CV fold", "imputation_strategy", "sampling_strategy", "scaling_strategy", "criterion", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "class_weight"])
inc_hp_gb = pd.DataFrame(columns=["Dataset ID", "CV fold", "imputation_strategy", "sampling_strategy", "scaling_strategy", "loss", "learning_rate", "criterion", "min_samples_split", "min_samples_leaf", "max_depth"])
inc_hp_svm = pd.DataFrame(columns=["Dataset ID", "CV fold", "imputation_strategy", "sampling_strategy", "scaling_strategy", "C", "kernel", "shrinking", "tol", "class_weight"])

for id in data_ids:
    for cv_fold in range(1,4):

        path_cv = output_path + '/dataset_{}_cv_{}'.format(id,cv_fold)

        files_in_dir = np.array(os.listdir(path_cv))

        inc_files = np.array(["incumbent" in file_in_dir for file_in_dir in files_in_dir])

        inc_files = files_in_dir[inc_files]

        for file_name in inc_files:
            hp_dict = json.load(open(path_cv + "/" + file_name))["config"]
            hp_dict["Dataset ID"] = id
            hp_dict["CV fold"] = cv_fold

            warnings.simplefilter(action='ignore', category=FutureWarning)
            if "rf" in file_name:
                inc_hp_rf = pd.concat([inc_hp_rf, pd.DataFrame([hp_dict])], ignore_index=True)
            elif "gb" in file_name:
                inc_hp_gb = pd.concat([inc_hp_gb, pd.DataFrame([hp_dict])], ignore_index=True)
            elif "svm" in file_name:
                inc_hp_svm = pd.concat([inc_hp_svm, pd.DataFrame([hp_dict])], ignore_index=True)

if not os.path.exists(output_path + "/incumbents_tables"):
    os.makedirs(output_path + "/incumbents_tables")

inc_hp_rf.rename(columns={"imputation_strategy": "Imputer","sampling_strategy": "Sampler", "scaling_strategy": "Scaler", "criterion": "Criterion", "max_depth": "Max depth", "min_samples_split": "Min samples per split", "min_samples_leaf": "Min samples per leaf", "max_features": "Max features", "class_weight": "Class weight"}, inplace=True)
inc_hp_rf.style.hide(axis="index").format(precision=3).to_latex(output_path + "/incumbents_tables/incumbents_rf.txt")

inc_hp_gb.rename(columns={"imputation_strategy": "Imputer","sampling_strategy": "Sampler", "scaling_strategy": "Scaler", "loss": "Loss", "learning_rate": "Learning rate", "criterion": "Criterion", "min_samples_split": "Min samples per split",  "min_samples_leaf": "Min samples per leaf", "max_depth": "Max depth"}, inplace=True)
inc_hp_gb.style.hide(axis="index").format(precision=3).to_latex(output_path + "/incumbents_tables/incumbents_gb.txt")

inc_hp_svm.rename(columns={"imputation_strategy": "Imputer","sampling_strategy": "Sampler", "scaling_strategy": "Scaler", "kernel": "Kernel", "shrinking": "Shrinking", "tol": "Tolerance", "class_weight": "Class weight"}, inplace=True)
inc_hp_svm.style.hide(axis="index").format(precision=3).to_latex(output_path + "/incumbents_tables/incumbents_svm.txt")
