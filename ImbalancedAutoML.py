import os
import time
from typing import Union, List, Dict
from datetime import datetime
import pickle

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.ensemble._base import BaseEnsemble
from sklearn.model_selection import KFold, cross_val_score
from sklearn import impute, ensemble, svm
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

from imblearn import under_sampling, over_sampling, combine
from imblearn.pipeline import Pipeline

from dehb import DEHB
from ConfigSpace import Configuration, ConfigurationSpace, Categorical, Float, Integer, UniformIntegerHyperparameter

from configuration import scoring


class ImbalancedAutoML(ClassifierMixin, BaseEnsemble):
    def __init__(self,
                 min_budget: Float = 2, 
                 max_budget: Float = 50, 
                 total_cost: Float = 3600,
                 random_state: int = 42
                 ) -> None:
        super().__init__()
        
        # Check how to make no sampling_strategy possible
        self.sampling_strategies = {
                "SMOTE": over_sampling.SMOTE(random_state=random_state),
                # "ADASYN": over_sampling.ADASYN(sampling_strategy="minority", random_state=random_state), # Still throws error as "No samples will be generated with the provided ratio settings."
                "Tomek": under_sampling.TomekLinks(),
                # "NCR": under_sampling.NeighbourhoodCleaningRule()
                "SMOTETomek": combine.SMOTETomek(random_state=random_state),
                "None": FunctionTransformer()
            }
        
        # Check whether makes sense for categorical features
        self.imputation_strategies = {
                "Simple": impute.SimpleImputer(strategy="median"),
                "KNN": impute.KNNImputer()
            }
        
        self.scaling_strategy = {
            "True": StandardScaler(),
            "False": FunctionTransformer()
        }

        self.model_dict = {"rf": ensemble.RandomForestClassifier, "gb": ensemble.GradientBoostingClassifier, "svm": svm.SVC}
        
        self.random_state = random_state

        self.cv = KFold(n_splits=4, shuffle=True, random_state=random_state)

        self.min_budget = min_budget
        self.max_budget = max_budget
        self.total_cost = total_cost

        self.model_cost_dict = {"rf": self.total_cost/3, "gb": self.total_cost/3, "svm": self.total_cost/3}

        self.dehb_objects = []
        self.trajectories = []
        self.runtimes = []
        self.histories = []
        self.model = []

    def create_search_space(self, model_name, seed=42):
        
        assert model_name in ["rf", "gb", "svm"]

        cs = ConfigurationSpace(seed)

        # https://automl.github.io/ConfigSpace/main/api/hyperparameters.html#ConfigSpace.hyperparameters.UniformIntegerHyperparameter

        sampling_strategy = Categorical(
            "sampling_strategy", items=["SMOTE", "Tomek", "SMOTETomek", "None"] # "ADASYN", "NCR", "None"
        )

        imputation_strategy = Categorical(
            "imputation_strategy", items=["Simple", "KNN"] # "Iterative", 
        )

        scaling_strategy = Categorical(
            "scaling_strategy", items=["True", "False"]
        )

        if model_name == "rf":
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            n_estimators = UniformIntegerHyperparameter(
                'n_estimators', lower=50, upper=500, default_value=100, log=False, q=50
            )

            criterion = Categorical(
                "criterion", items=["gini", "entropy", "log_loss"], default="gini"
            )

            max_depth = Integer(
                'max_depth', (5, 15), default=10, log=False
            )

            min_samples_split = Integer(
                'min_samples_split', (1, 64), default=2, log=True
            )

            min_samples_leaf = Integer(
                'min_samples_leaf', (1, 16), default=1, log=True
            )

            # Check whether makes sense
            max_features = Float(
                'max_features', (0.1, 0.9), default=0.5, log=False
            )

            class_weight = Categorical(
                "class_weight", items=["balanced", "balanced_subsample", "None"], default="None"
            )
        
            cs.add_hyperparameters([sampling_strategy, imputation_strategy, scaling_strategy, n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, class_weight])

        elif model_name == "gb":
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

            loss = Categorical(
                "loss", items=["exponential", "log_loss"]
            )

            learning_rate = Float(
                'learning_rate', (0.01, 0.2), default=0.1, log=True
            )

            n_estimators = Integer(
                'n_estimators', (50, 500), default=100, log=False, q = 50
            )

            criterion = Categorical(
                "criterion", items=["friedman_mse", "squared_error"]
            )

            min_samples_split = Integer(
                'min_samples_split', (1, 32), default=2, log=True
            )

            min_samples_leaf = Integer(
                'min_samples_leaf', (1, 32), default=1, log=True
            )

            cs.add_hyperparameters([sampling_strategy, imputation_strategy, scaling_strategy, loss, learning_rate, n_estimators, criterion, min_samples_split, min_samples_leaf])


        elif model_name == "svm":
            # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

            C = Float(
                "C", (0.1, 10), default=1, log=True
            )

            kernel = Categorical(
                "kernel", items=["linear", "poly", "rbf", "sigmoid"]
            )

            shrinking = Categorical(
                "shrinking", items=[True, False]
            )

            tol = Float(
                "tol", (1e-4, 1e-2), default=1e-3, log=True
            )

            class_weight = Categorical(
                "class_weight", items=["balanced", "None"]
            )

            cs.add_hyperparameters([sampling_strategy, imputation_strategy, scaling_strategy, C, kernel, shrinking, tol, class_weight])

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

            config_dict = config.get_dictionary()

            imputation_strategy = self.imputation_strategies[config["imputation_strategy"]]
            config_dict.pop("imputation_strategy")
            sampling_strategy = self.sampling_strategies[config["sampling_strategy"]]
            config_dict.pop("sampling_strategy")
            scaling_strategy = self.scaling_strategy[config["scaling_strategy"]]
            config_dict.pop("scaling_strategy")

            if kwargs["model_name"] in ["rf", "svm"]:
                if config_dict["class_weight"] == "None":
                    config_dict["class_weight"] = None

            model = Pipeline(
                steps=[
                    ("imputer", imputation_strategy),
                    ("imb_sampler", sampling_strategy),
                    ("round", self.column_transformer),
                    ("scaler", scaling_strategy), # Check how to make optional
                    ("estimator", kwargs["model"](**config_dict)),
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
            save_optim_output: bool = True,
            output_path: str = "results",
            output_name: Union[str, None] = None,
            seed: int = 42
            ):

        self.y_classes = np.unique(y)

        categorical_features = []

        for col in range(X.shape[1]):
            if np.all(np.logical_or(X[:,col] % 1 == 0, np.isnan(X[:,col] % 1))):
                categorical_features += [col]

        self.column_transformer = ColumnTransformer(
                [("round", FunctionTransformer(np.round), categorical_features),
                 ("identity", FunctionTransformer(), list(set(range(X.shape[1])) - set(categorical_features)))])

        best_inc_score = 0

        model_number = 0
        
        results_path = output_path + "/dataset_" + output_name + "_time_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        
        # This only works for one run
        for i in range(1,4):
            results_path = output_path + "/dataset_" + output_name + "_cv_" + str(i)
            if not os.path.exists(results_path):
                os.makedirs(results_path)
                break

        for model_name, model in self.model_dict.items():

            cs = self.create_search_space(model_name, seed)

            dimensions = len(cs.get_hyperparameters())

            dehb = DEHB(
                f=self.target_function, 
                cs=cs, 
                dimensions=dimensions, 
                min_budget=self.min_budget, 
                max_budget=self.max_budget,
                n_workers=1, # os.cpu_count()-1 throws error, check why
                output_path=results_path
            )

            # dehb.reset() # Necessary?
            trajectory, runtime, history = dehb.run(
                total_cost=self.model_cost_dict[model_name],
                verbose=verbose,
                save_intermediate=save_intermediate,
                # parameters expected as **kwargs in target_function is passed here
                seed=seed,
                train_X=X,
                train_y=y,
                model_name=model_name,
                model=model,
                max_budget=dehb.max_budget,
                name=output_name + "_model_" + str(model_name) + "_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") if output_name else output_name
            )

            self.dehb_objects.append(dehb)
            self.trajectories.append(trajectory)
            self.runtimes.append(runtime)
            self.histories.append(history)

            # Save dehb_objects, trajectories, runtimes and histories

            if  dehb.inc_score < best_inc_score:
                self.best_model = model_name
                best_inc_score = dehb.inc_score
        
            best_config = dehb.vector_to_configspace(dehb.inc_config).get_dictionary()

            imputation_strategy = self.imputation_strategies[best_config["imputation_strategy"]]
            best_config.pop("imputation_strategy")              
            sampling_strategy = self.sampling_strategies[best_config["sampling_strategy"]]
            best_config.pop("sampling_strategy")
            scaling_strategy = self.scaling_strategy[best_config["scaling_strategy"]]
            best_config.pop("scaling_strategy")

            if model_name in ["rf", "svm"]:
                if best_config["class_weight"] == "None":
                    best_config["class_weight"] = None

            self.model.append(Pipeline(
                steps=[
                    ("imputer", imputation_strategy),
                    ("imb_sampler", sampling_strategy),
                    ("round", self.column_transformer),
                    ("scaler", scaling_strategy),
                    ("estimator", model(**best_config)),
                ]
            ))

            self.model[model_number].fit(X, y)
            self.model[model_number].steps.pop(1)
            model_number += 1

        if save_optim_output:
            pickle.dump(self.dehb_objects, open(results_path + "/dehb_objects.pkl", 'wb'))
            pickle.dump(self.trajectories, open(results_path + "/trajectories.pkl", 'wb'))
            pickle.dump(self.runtimes, open(results_path + "/runtimes.pkl", 'wb'))
            pickle.dump(self.model, open(results_path + "/model.pkl", 'wb'))

        return self

    def predict(self, X=None, type: any = "ensemble"): # Ensemble or best?
        assert type in ["best", "ensemble", "rf", "gb", "svm"], "Invalid prediction type."
        model_to_number_dict = {"rf": 0, "gb": 1, "svm": 2} # Improve
        
        if type == "best":
            return self.model[model_to_number_dict[self.best_model]].predict(X)
        elif type == "ensemble":
            predictions = []
            for model_number in range(3):
                predictions.append(self.model[model_number].predict(X))
            predictions = np.array(predictions) == self.y_classes[0]
            prediction = (predictions.sum(axis=0) / predictions.shape[0]) >= 0.5
            return np.where(prediction, self.y_classes[0], self.y_classes[1])
        else:
            return self.model[model_to_number_dict[type]].predict(X)
    
    def get_dehb(self):
        return self.dehb_objects
    
    def get_trajectories(self):
        return self.trajectories
    
    def get_runtimes(self):
        return self.runtimes
    
    def get_histories(self):
        return self.histories
    

