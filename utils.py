import os
import numpy as np

# McNemar test taken from the solutions of the exercise sheets of the course.
def McNemar_test(labels, prediction_1, prediction_2):
    """
    :param labels: the ground truth labels
    :param prediction_1: the prediction results from model 1
    :param prediction_2:  the prediction results from model 2
    :return: the test statistic chi2_Mc
    """
    correct_model1 = labels == prediction_1
    correct_model2 = labels == prediction_2

    # A = sum(correct_model1 & correct_model2)
    B = sum(correct_model1 & ~correct_model2)
    C = sum(~correct_model1 & correct_model2)
    # D = sum(~correct_model1 & ~correct_model2)

    chi2_Mc = ((abs(B - C) - 1) ** 2) / (B + C)

    return chi2_Mc

def delete_large_file(output_path: str):
    """Deletes large and files not needed anymore like logs and histories.

    Args:
        output_path (str): Output path where results of the run are stored.
    """
    # Collect all folders from output path, which contain the results of each CV for each dataset
    folders_in_dir = np.array(os.listdir(output_path))
    dataset_folders = np.array(["dataset" in folder_in_dir for folder_in_dir in folders_in_dir])
    dataset_folders = folders_in_dir[dataset_folders]

    # Delete the log files and history from each folder, since they are large and not needed for further analysis.
    for dataset_folder in dataset_folders:
        files_in_folder = np.array(os.listdir(output_path + "/" + dataset_folder))
        log_files = np.array(["dehb" in file_in_folder for file_in_folder in files_in_folder])
        log_files = files_in_folder[log_files]
        history_files = np.array(["history" in file_in_folder for file_in_folder in files_in_folder])
        history_files = files_in_folder[history_files]
        for log_file in log_files:
            os.remove(output_path + "/" + dataset_folder + "/" + log_file)
        for history_file in history_files:
            os.remove(output_path + "/" + dataset_folder + "/" + history_file)