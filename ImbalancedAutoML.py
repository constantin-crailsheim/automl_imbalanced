import os
import time
from typing import Union, List, Dict
from datetime import datetime
import pickle
import warnings

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
from ConfigSpace import Configuration, ConfigurationSpace, Categorical, Float, Integer

from configuration import scoring


class ImbalancedAutoML(ClassifierMixin, BaseEnsemble):
    def __init__(self,
                 total_cost: Float = 3600,
                 random_state: int = 42
                 ) -> None:
        """
        Initializes ImbalancedAutoML object

        Args:
            total_cost (Float, optional): Total cost in seconds. Defaults to 3600.
            random_state (int, optional): Seed to be used as random state for some models. Defaults to 42.
        """
        super().__init__()
        
        self.sampling_strategies = {
                "SMOTE": over_sampling.SMOTE(random_state=random_state),
                "Tomek": under_sampling.TomekLinks(),
                "SMOTETomek": combine.SMOTETomek(random_state=random_state),
                "None": FunctionTransformer()
            }
        
        self.imputation_strategies = {
                "Simple": impute.SimpleImputer(strategy="median"),
                "KNN": impute.KNNImputer()
            }
        
        self.scaling_strategy = {
            "True": StandardScaler(),
            "False": FunctionTransformer()
        }

        self.model_dict = {"rf": ensemble.RandomForestClassifier, "gb": ensemble.GradientBoostingClassifier, "svm": svm.SVC}
        self.model_to_number_dict = {"rf": 0, "gb": 1, "svm": 2} 
        self.model_cost_dict = {"rf": total_cost*0.4, "gb": total_cost*0.4, "svm": total_cost*0.2} # Check the best split between costs
      
        self.random_state = random_state

        self.cv = KFold(n_splits=4, shuffle=True, random_state=random_state)

        self.min_budget = {"rf": 10, "gb": 10, "svm": 500} # Check sensible max_iter: Alternative 10 to 270
        self.max_budget = {"rf": 270, "gb": 270, "svm": 13500} # Check sensible max_iter
        self.total_cost = total_cost

        self.dehb_objects = []
        self.trajectories = []
        self.runtimes = []
        self.histories = []
        self.model = []

    def create_search_space(self, model_name: str, seed: int = 42):
        """
        Create a search space for DEHB in the logic of ConfigSpace.

        Args:
            model_name (str): Acronym of model name (should be "rf", "gb" or "svm").
            seed (int): Seed to initialize the ConfigurationSpace. Defaults to 42.

        Returns:
            _type_: ConfigurationSpace for model
        """
        assert model_name in ["rf", "gb", "svm"]

        cs = ConfigurationSpace(seed)

        # Set the choices for pre-processing  as categorical hyparparameter
        sampling_strategy = Categorical(
            "sampling_strategy", items=["SMOTE", "Tomek", "SMOTETomek", "None"]
        )
        imputation_strategy = Categorical(
            "imputation_strategy", items=["Simple", "KNN"]
        )
        scaling_strategy = Categorical(
            "scaling_strategy", items=["True", "False"]
        )

        if model_name == "rf":
            # Set the search space of the for the RandomForestClassifier
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

            criterion = Categorical(
                "criterion", items=["gini", "entropy", "log_loss"], default="gini"
            )
            max_depth = Integer(
                'max_depth', (5, 25), default=15, log=False
            )
            min_samples_split = Integer(
                'min_samples_split', (1, 32), default=2, log=True
            )
            min_samples_leaf = Integer(
                'min_samples_leaf', (1, 16), default=1, log=True
            )
            max_features = Float(
                'max_features', (0.1, 0.9), default=0.5, log=False # Check whether makes sense
            )
            class_weight = Categorical(
                "class_weight", items=["balanced", "balanced_subsample", "None"], default="None"
            )
        
            cs.add_hyperparameters([sampling_strategy, imputation_strategy, scaling_strategy, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, class_weight])

        elif model_name == "gb":
            # Set the search space of the for the GradientBoostingClassifier
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

            loss = Categorical(
                "loss", items=["exponential", "log_loss"], default="log_loss"
            )
            learning_rate = Float(
                'learning_rate', (0.01, 1), default=0.1, log=True
            )
            criterion = Categorical(
                "criterion", items=["friedman_mse", "squared_error"], default="friedman_mse"
            )
            min_samples_split = Integer(
                'min_samples_split', (2, 32), default=2, log=True
            )
            min_samples_leaf = Integer(
                'min_samples_leaf', (1, 16), default=1, log=True
            )
            max_depth = Integer(
                'max_depth', (2, 15), default=3, log=False
            )

            cs.add_hyperparameters([sampling_strategy, imputation_strategy, scaling_strategy, loss, learning_rate, criterion, min_samples_split, min_samples_leaf, max_depth])


        elif model_name == "svm":
            # Set the search space of the for the SVC
            # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

            C = Float(
                "C", (0.1, 10), default=1.0, log=True
            )
            kernel = Categorical(
                "kernel", items=["linear", "poly", "rbf", "sigmoid"], default="rbf"
            )
            shrinking = Categorical(
                "shrinking", items=[True, False]
            )
            tol = Float(
                "tol", (1e-4, 1e-2), default=1e-3, log=True
            )
            class_weight = Categorical(
                "class_weight", items=["balanced", "None"], default=None
            )

            cs.add_hyperparameters([sampling_strategy, imputation_strategy, scaling_strategy, C, kernel, shrinking, tol, class_weight])

        return cs
    
    def target_function(self,
            config: Union[Configuration, List, np.array], 
            budget: Union[int, float] = None,
            **kwargs
        ) -> Dict:
        """
        Target function to be optimized by DEHB.

        Args:
            config (Union[Configuration, List, np.array]): Hyperparameter configurations.
            budget (Union[int, float], optional): Budget of current fidelity. Defaults to None.

        Returns:
            Dict: Fitness as negative accuracy, cost as evaluation time and further info of test accuracy and budget.
        """
            
        train_X = kwargs["train_X"]
        train_y = kwargs["train_y"]
        
        start = time.time()

        # Add the budget to the config_dict passed to the function. 
        config_dict = config.get_dictionary()
        if kwargs["model_name"] in ["rf", "gb"]:
            config_dict["n_estimators"] = int(budget)
        elif kwargs["model_name"] in ["svm"]:
            config_dict["max_iter"] = int(budget)

        model = self.make_pipeline(config_dict, kwargs["model"], kwargs["model_name"])

        warnings.filterwarnings('ignore', 'Solver terminated early.*') # Ignores max_iter warning

        score = np.mean(cross_val_score(model, train_X, train_y, scoring=scoring, cv=self.cv))

        cost = time.time() - start

        # print("Model {} optimized with budget {} at cost {}".format(kwargs["model_name"], int(budget), cost))
        
        result = {
            "fitness": -score, 
            "cost": cost,
            "info": {
                "test_score": score,
                "budget": budget
            }
        }
        
        return result

    def make_pipeline(self, config_dict: Dict, model, model_name: str):
        """
        Builds pipeline based on current hyperparameter configuration.

        Args:
            config_dict (Dict): Dictionary of hyperparameter configuration.
            model (_type_): Model used as last element in pipeline. 
            model_name (str): Acronym of model name (should be "rf", "gb" or "svm").

        Returns:
            _type_: Pipeline of imputer, imbalanced sampler, rounding, scaler and model estimator. 
        """

        imputation_strategy = self.imputation_strategies[config_dict["imputation_strategy"]]
        config_dict.pop("imputation_strategy")
        sampling_strategy = self.sampling_strategies[config_dict["sampling_strategy"]]
        config_dict.pop("sampling_strategy")
        scaling_strategy = self.scaling_strategy[config_dict["scaling_strategy"]]
        config_dict.pop("scaling_strategy")

        if model_name in ["rf", "svm"]:
            if config_dict["class_weight"] == "None":
                config_dict["class_weight"] = None
        
        pipeline = Pipeline(
                steps=[
                    ("imputer", imputation_strategy),
                    ("imb_sampler", sampling_strategy),
                    ("round", self.column_transformer),
                    ("scaler", scaling_strategy),
                    ("estimator", model(**config_dict)),
                ]
            )

        return pipeline

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
        """
        Fits the AutoML system.

        Args:
            X (_type_, optional): Array of features. Defaults to None.
            y (_type_, optional): Array of targets. Defaults to None.
            verbose (bool, optional): Whether the DEHB optimizer should be verbose. Defaults to False.
            save_intermediate (bool, optional): Whether intermediate performance results should be saved. Defaults to True.
            save_optim_output (bool, optional): Whether the output of DEHB should be saved. Defaults to True.
            output_path (str, optional): The path to store the results. Defaults to "results".
            output_name (Union[str, None], optional): Name of output, should be set to id of dataset. Defaults to None.
            seed (int, optional): Seed to use for the optimization. Defaults to 42.

        Returns:
            _type_: self
        """

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
        
        # results_path = output_path + "/dataset_" + output_name + "_time_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        
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
                min_budget=self.min_budget[model_name], 
                max_budget=self.max_budget[model_name],
                n_workers=1, # os.cpu_count()-1 throws error, check why
                output_path=results_path,
                seed=seed
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

            if  dehb.inc_score < best_inc_score:
                self.best_model = model_name
                best_inc_score = dehb.inc_score
        
            best_config_dict = dehb.vector_to_configspace(dehb.inc_config).get_dictionary()
            if model_name in ["rf", "gb"]:
                best_config_dict["n_estimators"] = int(self.max_budget[model_name])
            elif model_name in ["svm"]:
                best_config_dict["max_iter"] = int(self.max_budget[model_name])

            self.model.append(self.make_pipeline(best_config_dict, model, model_name))

            # warnings.resetwarnings()

            self.model[model_number].fit(X, y)

            self.model[model_number].steps.pop(1)

            model_number += 1

        # Save dehb_objects, trajectories, runtimes and models.
        if save_optim_output:
            pickle.dump(self.dehb_objects, open(results_path + "/dehb_objects.pkl", 'wb'))
            pickle.dump(self.trajectories, open(results_path + "/trajectories.pkl", 'wb'))
            pickle.dump(self.runtimes, open(results_path + "/runtimes.pkl", 'wb'))
            pickle.dump(self.model, open(results_path + "/model.pkl", 'wb'))

        return self

    def predict(self, X=None, type: str = "stacking"):
        """
        Predict the target based on features.

        Args:
            X (_type_, optional): Array of features. Defaults to None.
            type (str, optional): Which models of the AutoML to use for prediction. Defaults to "stacking".

        Returns:
            _type_: Predictions of target.
        """

        assert type in ["best", "stacking", "rf", "gb", "svm"], "Invalid prediction type."
        
        if type == "best":
            return self.model[self.model_to_number_dict[self.best_model]].predict(X)
        elif type == "stacking":
            predictions = []
            for model_number in range(3):
                predictions.append(self.model[model_number].predict(X))
            predictions = np.array(predictions) == self.y_classes[0]
            prediction = (predictions.sum(axis=0) / predictions.shape[0]) >= 0.5
            return np.where(prediction, self.y_classes[0], self.y_classes[1])
        else:
            return self.model[self.model_to_number_dict[type]].predict(X)
    
    def get_dehb(self):
        """
        Allows to access the DEHB objects for all three models.

        Returns:
            List: List of DEHB objects.
        """
        return self.dehb_objects
    
    def get_trajectories(self):
        """
        Allows to access the trajectories for all three models.

        Returns:
            List: List of trajectories.
        """
        return self.trajectories
    
    def get_runtimes(self):
        """
        Allows to access the runtimes for all three models.

        Returns:
            List: List of runtimes.
        """
        return self.runtimes
    
    def get_histories(self):
        """
        Allows to access the histories for all three models.

        Returns:
            List: List of histories.
        """
        return self.histories
    

