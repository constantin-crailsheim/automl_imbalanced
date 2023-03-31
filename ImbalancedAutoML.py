import os
import time
from typing import Union, List, Dict
from datetime import datetime
import pickle
import warnings

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.ensemble._base import BaseEnsemble
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import impute, ensemble, svm, linear_model
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

        # Dicts of model objects and allocated training cost per algorithm
        self.model_dict = {"rf": ensemble.RandomForestClassifier, "gb": ensemble.GradientBoostingClassifier, "svm": svm.SVC}
        self.model_cost_dict = {"rf": total_cost*0.4, "gb": total_cost*0.4, "svm": total_cost*0.2} # Check the best split between costs
      
        self.random_state = random_state

        # Set min and max budget of the fidelities for each algorithm.
        # For random forest and gradient boosting, the budget is the number of trees.
        # For SVM, the budget is the maximum number of iterations.
        self.min_budget = {"rf": 10, "gb": 10, "svm": 500}
        self.max_budget = {"rf": 270, "gb": 270, "svm": 13500}

        self.total_cost = total_cost

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
            # Set the search space for the RandomForestClassifier
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
            # Set the search space for the GradientBoostingClassifier
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
            # Set the search space for the SVC
            # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

            C = Float(
                "C", (0.1, 10), default=1.0, log=True
            )
            kernel = Categorical(
                "kernel", items=["linear", "poly", "rbf", "sigmoid"], default="rbf"
            )
            shrinking = Categorical(
                "shrinking", items=[True, False], default=True
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
            Dict: Fitness as negative accuracy, cost as evaluation time and further info of budget.
        """
            
        train_X = kwargs["train_X"]
        train_y = kwargs["train_y"]
        
        start = time.time()

        # Add the current budget of the fidelity to the config_dict passed to the function. 
        config_dict = config.get_dictionary()
        if kwargs["model_name"] in ["rf", "gb"]:
            config_dict["n_estimators"] = int(budget)
        elif kwargs["model_name"] in ["svm"]:
            config_dict["max_iter"] = int(budget)
        
        # Make pipeline for current model based on current hyperparameter configuration
        model = self.make_pipeline(config_dict, kwargs["model"], kwargs["model_name"])

        # Ignores warning for SVM in case max_iter does not allow for complete convergence.
        warnings.filterwarnings('ignore', 'Solver terminated early.*')

        # Cross-validated balanced accuracy of model with current hyperparameter configuration
        # Initialise 4-fold cross-validation which ensures that share of targets is preserved in each fold.
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=self.random_state)
        score = np.mean(cross_val_score(model, train_X, train_y, scoring=scoring, cv=cv))

        cost = time.time() - start

        # print("Model {} optimized with budget {} at cost {}".format(kwargs["model_name"], int(budget), cost))
        
        result = {
            "fitness": -score, 
            "cost": cost,
            "info": {
                "budget": budget
            }
        }
        
        return result
    
    def make_column_transformer(self, X, features_dtypes):
        # Generates array of column indices of features that are integers.
        int_features = []
        if features_dtypes is not None:
            for col in range(len(features_dtypes)):
                if np.issubdtype(features_dtypes[col], np.integer):
                    int_features += [col]

        # Initialises function that rounds values of features that are integers after imputation and sampling
        column_transformer = ColumnTransformer(
                [("round", FunctionTransformer(np.round), int_features),
                 ("identity", FunctionTransformer(), list(set(range(X.shape[1])) - set(int_features)))])
    
        return column_transformer

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

        # Set random state for all models
        config_dict["random_state"] = self.random_state

        # Takes objects of imputer, sampler and scaler from the list of choices and removes hyperparameter from config_dict.
        imputation_strategy = self.imputation_strategies[config_dict["imputation_strategy"]]
        config_dict.pop("imputation_strategy")
        sampling_strategy = self.sampling_strategies[config_dict["sampling_strategy"]]
        config_dict.pop("sampling_strategy")
        scaling_strategy = self.scaling_strategy[config_dict["scaling_strategy"]]
        config_dict.pop("scaling_strategy")

        # Replaces string "None" with None for class_weight if chosen as hyperparameter.
        if model_name in ["rf", "svm"]:
            if config_dict["class_weight"] == "None":
                config_dict["class_weight"] = None
        
        # Creates pipeline for current hyperparameter configuration.
        pipeline = Pipeline(
                steps=[
                    ("imputer", imputation_strategy),
                    ("imb_sampler", sampling_strategy),
                    ("round",self.column_transformer),
                    ("scaler", scaling_strategy),
                    ("estimator", model(**config_dict)),
                ]
            )

        return pipeline

    def fit(self, 
            X=None, 
            y=None,
            features_dtypes = None,
            verbose: bool = False,
            save_intermediate: bool = False,
            save_history: bool = False,
            save_optim_output: bool = True,
            output_path: str = "results/run_x",
            output_name: Union[str, None] = None,
            cv_fold: int = None,
            seed: int = 42
            ):
        """
        Fits the AutoML system.

        Args:
            X (_type_, optional): Array of train features. Defaults to None.
            y (_type_, optional): Array of train targets. Defaults to None.
            verbose (bool, optional): Whether the DEHB optimizer should be verbose. Defaults to False.
            save_intermediate (bool, optional): Whether intermediate performance results should be saved. Defaults to False.
            save_intermediate (bool, optional): Whether history should be saved. Defaults to False.
            save_optim_output (bool, optional): Whether the output of DEHB should be saved. Defaults to True.
            output_path (str, optional): The path to store the results. Defaults to "results".
            output_name (Union[str, None], optional): Name of output, should be set to id of dataset. Defaults to None.
            seed (int, optional): Seed to use for the optimization. Defaults to 42.
        """

        # Initialize lists to store objects during optimization
        self.dehb_objects = []
        self.trajectories = []
        self.runtimes = []
        self.histories = []
        self.model = []

        # Stores the number of classed of the target.
        self.y_classes = np.unique(y)

        # Generates function that rounds observations that were integers.
        self.column_transformer = self.make_column_transformer(X, features_dtypes)

        # Makes new directories for each dataset and cross-validation fold in the results folder.
        # This is easier to access for evaluation than adding time stamps
        # Caution: This only work properly if the results folder is empty
        results_path = output_path + "/dataset_" + output_name + "_cv_" + str(cv_fold)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        else:
            print("Warning: This output path does already exist, you might overwrite exisiting results.")

        # Loop through all three models and tune hyperparameter for each
        for model_name, model in self.model_dict.items():
            
            # Creates a search space for the model with the corresponding hyperparameter
            cs = self.create_search_space(model_name, seed)

            # Checks how many hyperparameters are to be tuned.
            dimensions = len(cs.get_hyperparameters())

            # Initializse DEHB object with min and max budget for current algorithm.
            dehb = DEHB(
                f=self.target_function, 
                cs=cs, 
                dimensions=dimensions, 
                min_budget=self.min_budget[model_name], 
                max_budget=self.max_budget[model_name],
                n_workers=1,
                output_path=results_path,
                seed=seed
            )

            # Run optimization with total cost taken from corresponds dictionary
            trajectory, runtime, history = dehb.run(
                total_cost=self.model_cost_dict[model_name],
                verbose=verbose,
                save_intermediate=save_intermediate,
                save_history=save_history,
                # parameters expected as **kwargs in target_function is passed here
                seed=seed,
                train_X=X,
                train_y=y,
                model_name=model_name,
                model=model,
                max_budget=dehb.max_budget,
                name=output_name + "_model_" + str(model_name) + "_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") if output_name else output_name
            )

            # Keep DEHB object and outputs to use later for evaluation of algorithm. 
            self.dehb_objects.append(dehb)
            self.trajectories.append(trajectory)
            self.runtimes.append(runtime)
            self.histories.append(history)
            
            # Add the highest fidelity budget to the incumbent hyperparameter configuration for final optimization.
            best_config_dict = dehb.vector_to_configspace(dehb.inc_config).get_dictionary()
            if model_name in ["rf", "gb"]:
                best_config_dict["n_estimators"] = int(self.max_budget[model_name])
            elif model_name in ["svm"]:
                best_config_dict["max_iter"] = int(self.max_budget[model_name])

            # Append incumbent model to list used for voting classifier
            self.model.append((model_name, self.make_pipeline(best_config_dict, model, model_name)))
        
        # Initialize and fit voting classifier on incumbent models
        self.voting_classifier = ensemble.VotingClassifier(self.model)
        self.voting_classifier.fit(X, y)

        # Remove sampler from pipeline to be used for testing.
        for model_number in range(3):
            self.model[model_number][1].steps.pop(1)

        # Save trajectories and runtimes. DEHB and voting classifier objects can be saved optionally.
        if save_optim_output:
            # pickle.dump(self.dehb_objects, open(results_path + "/dehb_objects.pkl", 'wb'))
            pickle.dump(self.trajectories, open(results_path + "/trajectories.pkl", 'wb'))
            pickle.dump(self.runtimes, open(results_path + "/runtimes.pkl", 'wb'))
            # pickle.dump(self.voting_classifier, open(results_path + "/voting_classifier.pkl", 'wb'))

        return self
    
    def fit_with_hp(self, X=None, y=None, hp_dict: Dict= None, features_dtypes=None):
        """
        Fits the voting classifier with an ensemble of RandomForestClassifer, GradientBoostingClassifier
        and SVC with given hyperparameter.

        Args:
            X (_type_, optional): Array of train features. Defaults to None.
            y (_type_, optional): Array of train targets. Defaults to None.
            hp_dict (Dict, optional): Dict of length 3 containing dicts of hyperparameter for each model. Defaults to None.
            features_dtypes (_type_, optional): Datatypes of all features. Defaults to None.
        """
        
        # Add the maximum budget to the hyperparameter dict for the pipeline of each model
        for model_name in hp_dict.keys():
            if model_name in ["rf", "gb"]:
                hp_dict[model_name]["n_estimators"] = int(self.max_budget[model_name])
            elif model_name in ["svm"]:
                hp_dict[model_name]["max_iter"] = int(self.max_budget[model_name])

        # Initialize function that rounds imputed or oversampled integer features
        self.column_transformer = self.make_column_transformer(X, features_dtypes)

        # Create list of pipeline for each model
        self.model = []
        for model_name, model in self.model_dict.items():
            self.model.append((model_name, self.make_pipeline(hp_dict[model_name], model, model_name)))
        
        # Initialize and fit voting classifier for given hyperparameter
        self.voting_classifier = ensemble.VotingClassifier(self.model)
        self.voting_classifier.fit(X, y)

        return self

    def predict(self, X=None):
        """
        Predict the target based on features.

        Args:
            X (_type_, optional): Array of features. Defaults to None.

        Returns:
            _type_: Predictions of target.
        """
        return self.voting_classifier.predict(X)
    
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
    

