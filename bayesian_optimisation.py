import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
from lightgbm import LGBMRegressor
from functools import partial
from hyperopt import fmin, tpe, Trials

class BayesianOptimisationLgbm:
    """
    A class to perform Bayesian Optimization for LightGBM using time series data.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    training_days (int): Number of training days.
    startdate_val (datetime.date): Start date for validation.
    validation_days (int): Number of validation days.
    model_name (str): The model name.
    target_name (str): The target variable name.
    covariate_names (list): List of covariate names.
    categorical_covariate_names (list): List of discrete covariate names.
    scale_data_flag (bool): Flag to scale data.
    n_splits (int): Number of splits for cross-validation.
    param_space (dict): Hyperparameter search space.
    max_evals (int): Maximum number of evaluations.
    int_parameters (list): List of integer parameters.
    option_parameters (dict): Dictionary of option parameters.
    """
    def __init__(self, df, training_days, startdate_val, validation_days,  model_name, target_name, covariate_names,
                 categorical_covariate_names, n_splits, param_space, max_evals, int_parameters,
                 option_parameters):

        self.df = df
        self.training_days = training_days
        self.startdate_val = startdate_val
        self.validation_days = validation_days
        self.model_name = model_name
        self.target_name = target_name
        self.covariate_names = covariate_names
        self.categorical_covariate_names = categorical_covariate_names
        self.n_splits = n_splits
        self.param_space = param_space
        self.max_evals = max_evals
        self.int_parameters = int_parameters
        self.option_parameters = option_parameters
        self.filenames = self.create_file_paths(f"{self.model_name}_{self.training_days}d")

    def create_file_paths(self, model_name):
        """
        Creates file paths for saving hyperparameters.

        Returns:
        dict: A dictionary containing file paths.
        """
        return {'hyperparams': f".\\models\\hyperparams_{model_name}.joblib"}

    def get_opt_hyperparams(self):
        """
        Performs hyperparameter optimization using Bayesian Optimization.

        Returns:
        dict: Optimized hyperparameters.
        """
        datasets, covariate_names = self.split_train_val_data()
        ind_categorical_covariate_names = [
            i for i, name in enumerate(covariate_names) if name in self.categorical_covariate_names
        ]

        # partial function
        optimize_func_partial = partial(
            self.optimize,
            X=datasets['X'],
            y=datasets['y'],
            categorical_covariate_indices=ind_categorical_covariate_names
        )

        # initialize trials to keep logging information
        trials = Trials()

        hopt = fmin(
            fn=optimize_func_partial,
            space=self.param_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials
        )

        # Convert integer parameters to int
        hopt = self._convert_int_parameters(hopt)

        # Map option parameters
        hopt = self._map_option_parameters(hopt)

        self.save_model_cache(self.filenames['hyperparams'], hopt)

        return hopt

    def split_train_val_data(self):
        """
        Splits the dataframe into training and validation sets.

        Returns:
        tuple: A dictionary of datasets and a list of covariate names.
        """
        data_dates = self.create_train_val_dates()
        df_subset = self.subset_data(data_dates)
        covariate_names = self.get_covariate_names(df_subset)
        datasets = self.create_datasets(df_subset, covariate_names)
        return datasets, covariate_names

    def create_train_val_dates(self):
        """
        Creates the start and end dates for training and validation periods.

        Returns:
        dict: A dictionary containing the start and end dates.
        """
        enddate_val = self.startdate_val + timedelta(days=self.validation_days)
        enddate_train = self.startdate_val - timedelta(days=1)
        startdate_train = enddate_train - timedelta(days=self.training_days)

        return {
            'startdate_train':startdate_train,
            'enddate_train': enddate_train,
            'startdate_val':self.startdate_val,
            'enddate_val': enddate_val,
        }

    def subset_data(self, data_dates):
        """
        Subsets the dataframe to the training and testing date range.

        Parameters:
        data_dates (dict): A dictionary containing the start and end dates for training and testing periods.

        Returns:
        pd.DataFrame: A subset of the dataframe.
        """
        df_subset = self.df[(self.df['datetime'].dt.date >= data_dates['startdate_train']) & (
                self.df['datetime'].dt.date <= data_dates['enddate_val'])].copy()
        return df_subset

    def get_covariate_names(self, df):
        """
        Retrieves covariate names excluding datetime and target.

        Parameters:
        df_subset (pd.DataFrame): The subset of the dataframe.

        Returns:
        list: A list of covariate names.
        """
        return [c for c in df.columns if c not in ['datetime', self.target_name]]

    def create_datasets(self, df, covariate_names):
        """
        Creates Covariate and target datasets.

        Parameters:
        df (pd.DataFrame): dataframe.
        covariate_names (list): A list of covariate names.

        Returns:
        dict: A dictionary containing the datasets.
        """
        return{
            'X': df[covariate_names].values,
            'y': df[self.target_name].values
        }

    def optimize(self, params, X, y, categorical_covariate_indices):
        """
        Optimization function for hyperparameter tuning.

        Parameters:
        params (dict): Hyperparameters for the model.
        n_splits (int): Number of splits for cross-validation.
        training_days (int): Number of training days.
        validation_days (int): Number of validation days.
        X (np.ndarray): Input features.
        y (np.ndarray): Target variable.
        categorical_covariate_names (list): List of categorical covariate names.

        Returns:
        float: Average mean absolute error score.
        """
        model = LGBMRegressor(**params)
        scores = []

        tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.training_days * 24,
                               test_size=(self.validation_days * 24) // self.n_splits)

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fit_model = model.fit(X_train, y_train, categorical_feature=categorical_covariate_indices,
                                  eval_set=[(X_test, y_test)])
            preds = fit_model.predict(X_test)
            score = mean_absolute_error(y_test, preds)
            scores.append(score)
            del fit_model, X_train, X_test, y_train, y_test, score, preds

        score_avg = self.weighted_average(scores)
        return score_avg

    def _convert_int_parameters(self, dict_hopt):
        """
        Converts specified parameters in the hyperparameter dictionary to integers.

        Parameters:
        dict_hopt (dict): Hyperparameter dictionary.

        Returns:
        dict: Updated hyperparameter dictionary with specified parameters converted to integers.
        """
        for parameter in self.int_parameters:
            if parameter in dict_hopt:
                dict_hopt[parameter] = int(dict_hopt[parameter])
        return dict_hopt

    def _map_option_parameters(self, dict_hopt):
        """
        Maps specified parameters in the hyperparameter dictionary to their corresponding options.

        Parameters:
        dict_hopt (dict): Hyperparameter dictionary.

        Returns:
        dict: Updated hyperparameter dictionary with specified parameters mapped to their options.
        """
        for parameter, options in self.option_parameters.items():
            if parameter in dict_hopt:
                dict_hopt[parameter] = options[dict_hopt[parameter]]
        return dict_hopt

    def weighted_average(self, a):
        """
        Calculates a weighted average of a list.

        Parameters:
        a (list): List of values.

        Returns:
        float: Weighted average.
        """
        w = [1 / float(2 ** (len(a) + 1 - j)) for j in range(1, len(a) + 1)]
        return np.average(a, weights=w)

    def save_model_cache(self, filename, model_cache):
        """
        Saves the model cache to a file.

        Parameters:
        filename (str): The path to the file.
        model_cache (dict): The model cache to be saved.

        Returns:
        None
        """
        try:
            print(f"Saving {filename}")
            with open(filename, 'wb') as f:
                joblib.dump(model_cache, f)
        except Exception as e:
            print(f"Error saving model cache to {filename}: {e}")