# AutoML Project: Imbalanced data classification

This project was part of the AutoML course in the winter term 2022/23 at LMU Munich by [Janek Thomas](https://www.slds.stat.uni-muenchen.de/people/thomas/). The key objective was to tune an AutoML system that outperforms a random forest baseline. The report can be found [here](https://github.com/constantin-crailsheim/automl_imbalanced/blob/main/Report/AutoML%20(Report).pdf).

# Fild structure

The repo contains the following files and folder:
- `ImbalancedAutoML.py`: Contains ImbalancedAutoML class, which is the AutoML system used.
- `configuration.py`: Contains the measure and openml ids
- `benchmark.py`: File to run benchmarks of AutoML system vs. random forest baseline.
- `data.py`: A dataclass for downloading and storing OpenML tasks
- `utils.py`: Contains McNemar test and method to delete large files from results.
- `visualisation.py`: Notebook to generate tables and plots used for report.
- `requirements.txt`: Contains all package requirements needed.
- `results`: Folder where results, plots and tables are stored.
- `report`: Folder with report files.
 
# Setup 

## Environment and requirements

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
- `imbalanced-learn`: Imbalanced sampling methods that can be easily combines with sklearn.
- `dehb`: Optimization framework to tune hyperparameter of pipeline.
- `ConfigSpace`: Used to define and sample from search space. 

# Replication of results

- Set all desired configurations in `configuration.py`.
- Run `benchmark.py` to compare AutoML system against the baseline and store results.
- Run `visualisation.py` to generate tables and plots of the performance and incumbents. 
