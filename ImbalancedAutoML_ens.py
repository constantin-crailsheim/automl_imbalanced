
import time
from typing import Union, List, Dict
from datetime import datetime

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.ensemble._base import BaseEnsemble
from sklearn.model_selection import KFold, cross_val_score
from sklearn import impute, pipeline, tree, ensemble

from imblearn import under_sampling, over_sampling

from dehb import DEHB
from ConfigSpace import Configuration, ConfigurationSpace, Categorical, Float, Integer

from configuration import scoring


class ImbalancedAutoML(ClassifierMixin, BaseEnsemble):
    def __init__(self,
                 min_budget: Float = 2, 
                 max_budget: Float = 50, 
                 total_cost: Float = 3600
                 ) -> None:
        super().__init__()

        self.sampling_strategies = {
                "SMOTE": over_sampling.SMOTE(random_state=42),
                "Tomek links": under_sampling.TomekLinks()
            }

        self.cv = KFold(n_splits=3, shuffle=True, random_state=42)

        self.min_budget = min_budget
        self.max_budget = max_budget
        self.total_cost = total_cost

        self.dehb_objects = []
        self.trajectories = []
        self.runtimes = []
        self.histories = []
        self.model = []

    def create_search_space(self, seed=42):

        cs = ConfigurationSpace(seed)

        sampling_strategy = Categorical(
            "sampling_strategy", items=["SMOTE", "Tomek links", "None"]
        )

        n_estimators = Integer(
            'n_estimators', (50, 500), default=100, log=False, q = 50
        )

        criterion = Categorical(
            "criterion", items=["gini", "entropy", "log_loss"]
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
        min_samples_leaf = Integer(
            'min_samples_leaf', (1, 64), default=5, log=True
        )

        class_weight = Categorical(
            "class_weight", items=["balanced", "balanced_subsample", "None"]
        )

        cs.add_hyperparameters([sampling_strategy, n_estimators, criterion, max_depth, min_samples_split, max_features, min_samples_leaf, class_weight])

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
            if config["sampling_strategy"] != "None":
                train_X, train_y = self.sampling_strategies[config["sampling_strategy"]].fit_resample(train_X, train_y)
            config_dict = config.get_dictionary()
            config_dict.pop("sampling_strategy")
            if config_dict["class_weight"] == "None":
                config_dict["class_weight"] = None
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

    def fit(self, 
            X=None, 
            y=None,
            verbose: bool = False,
            save_intermediate: bool = True,
            number_restarts: int = 1,
            output_path: str = "results",
            output_name: Union[str, None] = None,
            seed: int = 42
            ):

        self.number_restarts = number_restarts
        self.y_classes = np.unique(y)

        best_inc_score = 0

        for restart_number in range(1,number_restarts+1):
            current_seed = seed*restart_number
            cs = self.create_search_space(current_seed)

            dimensions = len(cs.get_hyperparameters())

            dehb = DEHB(
                f=self.target_function, 
                cs=cs, 
                dimensions=dimensions, 
                min_budget=self.min_budget, 
                max_budget=self.max_budget,
                n_workers=1,
                output_path=output_path
            )

            trajectory, runtime, history = dehb.run(
                total_cost=self.total_cost/number_restarts,
                verbose=verbose,
                save_intermediate=save_intermediate,
                # parameters expected as **kwargs in target_function is passed here
                seed=current_seed,
                train_X=X,
                train_y=y,
                max_budget=dehb.max_budget,
                name=output_name + "_restart_" + str(restart_number) + "_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") if output_name else output_name
            )

            self.dehb_objects.append(dehb)
            self.trajectories.append(trajectory)
            self.runtimes.append(runtime)
            self.histories.append(history)

            if  dehb.inc_score < best_inc_score:
                self.best_restart_number = restart_number
                best_inc_score = dehb.inc_score
        
            best_config = dehb.vector_to_configspace(dehb.inc_config).get_dictionary()
            if best_config["sampling_strategy"] != "None":
                X, y = self.sampling_strategies[best_config["sampling_strategy"]].fit_resample(X, y)
            best_config.pop("sampling_strategy")
            if best_config["class_weight"] == "None":
                best_config["class_weight"] = None
            self.model.append(pipeline.Pipeline(
                steps=[
                    ("imputer", impute.SimpleImputer()),
                    ("estimator", ensemble.RandomForestClassifier(**best_config)),
                ]
            ))
            self.model[restart_number-1].fit(X, y)

        return self

    def predict(self, X=None, type: any = "best"):
        assert type in ["best", "ensemble"] + list(range(1,self.number_restarts+1)), "Invalid prediction type."

        if type == "best":
            return self.model[self.best_restart_number-1].predict(X)
        elif type == "ensemble":
            assert self.number_restarts % 2 == 1, "Number of restarts needs to be odd to allow majority voting."
            predictions = []
            for restart_number in range(self.number_restarts):
                predictions.append(self.model[restart_number].predict(X))
            predictions = np.array(predictions) == self.y_classes[0]
            prediction = (predictions.sum(axis=0) / predictions.shape[0]) >= 0.5
            return np.where(prediction, self.y_classes[0], self.y_classes[1])
        else:
            return self.model[type-1].predict(X)
    
    def get_dehb(self):
        return self.dehb_objects
    
    def get_trajectories(self):
        return self.trajectories
    
    def get_runtimes(self):
        return self.runtimes
    
    def get_histories(self):
        return self.histories
    

