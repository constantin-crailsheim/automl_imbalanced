
import time
from typing import Union, List, Dict

from sklearn.base import ClassifierMixin
from sklearn.model_selection import KFold, cross_val_score
from sklearn import impute, pipeline, tree, ensemble
from sklearn.metrics import balanced_accuracy_score

import numpy as np


from imblearn import under_sampling, over_sampling

from dehb import DEHB
from ConfigSpace import Configuration, ConfigurationSpace, Categorical, Float, Integer


from configuration import scoring


class ImbalancedAutoML(ClassifierMixin):
    def __init__(self) -> None:
        super().__init__()

        self.sampling_strategies = {
                "SMOTE": over_sampling.SMOTE(random_state=42),
                "Tomek links": under_sampling.TomekLinks()
            }

        self.cv = KFold(n_splits=2, shuffle=True, random_state=42)

        self.min_budget = 2

        self.max_budget = 50


    def create_search_space(self, seed=42):

        cs = ConfigurationSpace(seed)

        sampling_strategy = Categorical(
            "sampling_strategy", items=["SMOTE", "Tomek links"]
        )

        max_depth = Integer(
            'max_depth', (1, 15), default=2, log=False
        )
        min_samples_split = Integer(
            'min_samples_split', (2, 128), default=10, log=True
        )
        max_features = Float(
            'max_features', (0.1, 0.9), default=0.5, log=False
        )
        min_samples_lead = Integer(
            'min_samples_leaf', (1, 64), default=5, log=True
        )

        cs.add_hyperparameters([sampling_strategy, max_depth, min_samples_split, max_features, min_samples_lead])

        return cs
    
    def target_function(self,
            config: Union[Configuration, List, np.array], 
            budget: Union[int, float] = None,
            **kwargs
        ) -> Dict:
            """ Target/objective function to optimize
            
            Parameters
            ----------
            x : configuration that DEHB wants to evaluate
            budget : parameter determining cheaper evaluations
            
            Returns
            -------
            dict
            """

            train_X = kwargs["train_X"]
            train_y = kwargs["train_y"]
            
            start = time.time()
            train_X, train_y = self.sampling_strategies[config["sampling_strategy"]].fit_resample(train_X, train_y)
            config_dict = config.get_dictionary()
            config_dict.pop("sampling_strategy")
            model = pipeline.Pipeline(
                steps=[
                    ("imputer", impute.SimpleImputer()),
                    ("estimator", ensemble.RandomForestClassifier(**config_dict)),
                ]
            )
            score = np.mean(cross_val_score(model, train_X, train_y, scoring=scoring, cv=self.cv))
            cost = time.time() - start
            
            result = {
                "fitness": -score, 
                "cost": cost,
                "info": {
                    "test_score": score,
                    "budget": budget
                }
            }
            
            return result

    def fit(self, X=None, y=None, verbose=False, output_path="results", seed=42):

        cs = self.create_search_space(seed)

        dimensions = len(cs.get_hyperparameters())

        self.dehb = DEHB(
            f=self.target_function, 
            cs=cs, 
            dimensions=dimensions, 
            min_budget=self.min_budget, 
            max_budget=self.max_budget,
            n_workers=1,
            output_path=output_path
        )

        self.trajectory, self.runtime, self.history = self.dehb.run(
            total_cost=300,
            verbose=True,
            save_intermediate=True,
            # parameters expected as **kwargs in target_function is passed here
            seed=42,
            train_X=X,
            train_y=y,
            max_budget=self.dehb.max_budget
        )

        best_config = self.dehb.vector_to_configspace(self.dehb.inc_config).get_dictionary()

        X, y = self.sampling_strategies[best_config["sampling_strategy"]].fit_resample(X, y)
        best_config.pop("sampling_strategy")
        self.model = pipeline.Pipeline(
            steps=[
                ("imputer", impute.SimpleImputer()),
                ("estimator", ensemble.RandomForestClassifier(**best_config)),
            ]
        )
        self.model.fit(X, y)

        return self

    def predict(self, X=None):
        return self.model.predict(X)
    
    def get_dehb(self):
        return self.dehb
    
    def get_trajectory(self):
        return self.trajectory
    
    def get_runtime(self):
        return self.runtime
    
    def get_history(self):
        return self.history
    

