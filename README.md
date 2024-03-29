# AutoML Project: Imbalanced data classification

This project was part of the AutoML course in the winter term 2022/23 at LMU Munich by [Janek Thomas](https://www.slds.stat.uni-muenchen.de/people/thomas/). The key objective was to tune an AutoML system that outperforms a random forest baseline on 10 imbalanced classification tasks. The report can be found [here](https://github.com/constantin-crailsheim/automl_imbalanced/blob/main/report/AutoML%20(Report).pdf).

# File and folder structure

The repo contains the following files and folder:
- `report`: Folder with report files.
- `results`: Folder where results, plots and tables are stored. Contains results of run used for report. 
- `ImbalancedAutoML.py`: Contains ImbalancedAutoML class, which is the AutoML system used.
- `README.md`: Contains instructions about the repo.
- `benchmark.py`: Contains code to run benchmarks of AutoML system vs. random forest baseline.
- `configuration.py`: Contains configurations needed in other files.
- `data.py`: A dataclass for downloading and storing OpenML tasks.
- `project_automl_imbalanced.pdf`: Contains instructions about project.
- `requirements.txt`: Contains all package requirements needed.
- `utils.py`: Contains McNemar test and method to delete large files from results.
- `visualisation.py`: Contains code to generate tables and plots used for report.
 
# Setup 

First set up a conda environment:

```(bash)
conda create -n <env_name> python=3.11.0
conda activate <env_name>
```

Then install the recommended requirements with:

```(bash)
pip install -r requirements.txt
```

The used packages are the following:
- `openml`: Used to download the benchmarking datasets.
- `pandas`: Mostly used to generate dataframes to convert to Latex tables.
- `numpy`: Used for basic data operations and storage and is base for sklearn functions.
- `scikit-learn`: Elements of pipeline like preprocessing and models are taken from here.
- `imbalanced-learn`: Imbalanced sampling methods that can be easily combined with sklearn.
- `dehb`: Optimization framework to tune hyperparameters of pipeline.
- `ConfigSpace`: Used to define and sample from search space. 

# Replication of results

- Set all desired configurations in `configuration.py`.
- Run `benchmark.py` to compare AutoML system against the baseline and store results.
- Run `visualisation.py` to generate tables and plots of the performance and incumbents. 
