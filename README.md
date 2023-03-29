# AutoML Project: Imbalanced data classification

The repo contains the following files and folder:
- `ImbalancedAutoML.py`: Contains ImbalancedAutoML class, which is the AutoML system used.
- `configuration.py`: Contains the measure and openml ids
- `benchmark.py`: File to run benchmarks of AutoML system vs. random forest baseline.
- `data.py`: A dataclass for downloading and storing OpenML tasks
- `utils.py`: Contains McNemar test and method to delete large files from results.
- `visualisation.py`: Notebook to generate tables and plots used for report.
- `requirements.txt`: Contains all package requirements needed.

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

# Replication of results

- Set all desired configurations in `configuration.py`.
- Run `benchmark.py` to compare AutoML system against the baseline and store results.
- Run `visualisation.py` to generate tables and plots of the performance and incumbents. 
