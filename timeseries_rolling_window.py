import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from copy import deepcopy
from pygam import LinearGAM, s, f, te
from itertools import product
from models import Naive
import mlflow
import joblib

class RollingWindowBase:
    """
    A base class for implementing rolling window training and evaluation for time series forecasting models.
    """
    def __init__(self, df, y_val, startdate_val, validation_days, horizon_days_val, y_test, startdate_test, test_days,
                 horizon_days_test, gap_days, target_name, covariate_names, binary_covariate_names=None,
                 categorical_covariate_names=None):

        """
        Initializes the RollingWindowBase class with provided parameters.

        Parameters:
        - df (pd.DataFrame): The input dataframe containing the time series data.
        - y_val (np.ndarray): The target values for the validation set.
        - startdate_val (datetime.date): The start date for the validation period.
        - validation_days (int): The number of days in the validation period.
        - horizon_days_val (int): The forecasting horizon for the validation period.
        - y_test (np.ndarray): The target values for the test set.
        - startdate_test (datetime.date): The start date for the test period.
        - test_days (int): The number of days in the test period.
        - horizon_days_test (int): The forecasting horizon for the test period.
        - target_name (str): The name of the target variable in the dataframe.
        - covariate_names (list): The list of covariate names used for prediction.
        - binary_covariate_names (list, optional): The list of binary covariate names. Default is an empty list.
        - categorical_covariate_names (list, optional): The list of categorical covariate names. Default is an empty list.
        """

        self.df = df
        self.y_val = y_val
        self.startdate_val = startdate_val
        self.validation_days = validation_days
        self.horizon_days_val = horizon_days_val
        self.y_test = y_test
        self.startdate_test = startdate_test
        self.test_days = test_days
        self.horizon_days_test = horizon_days_test
        self.gap_days = gap_days
        self.target_name = target_name
        self.covariate_names = covariate_names
        self.binary_covariate_names = binary_covariate_names or []
        self.categorical_covariate_names = categorical_covariate_names or []
        self.dummy_feature_names = []
        self.dataset_names = ['train', 'val', 'test']
        self.initialise_attributes()

    def initialise_attributes(self):
        """
        Initializes model-related attributes with default values.
        """
        # Model configuration attributes
        self.scale_data_flag = False  # Whether to scale the data
        self.model_type = ""  # Type of the model (e.g., 'sklearn', 'arima')
        self.model_implementation = ""  # Specific implementation of the model
        self.model_name = ""  # Name of the model configuration
        self.model_info = {}  # Dictionary containing model-specific configurations and parameters
        self.model_params = {}  # Parameters for the model configuration

        # File path attributes
        self.filenames = {}  # Dictionary to store file paths for saving predictions and model cache

    def train_models(self, model_cache, save_predictions_flag):
        """
        Trains and evaluates models using a rolling window approach.

        Parameters:
        model_cache (dict): Dictionary containing model configurations and parameters.
        save_predictions_flag (dict): Dictionary of flag to determine whether to save predictions for the train,
        validation and test datasets.

        Returns:
        tuple: Contains two dictionaries:
            - predictions (dict): Predicted values for each model configuration.
            - results (dict): Evaluation metrics for each model configuration.
        """
        self.save_predictions_flag = save_predictions_flag
        predictions, results = {}, {}

        for model_type, model_info in model_cache.items():
            self.set_model_info(model_type, model_info)
            param_names = list(model_info['model_params'].keys())
            param_vals_combinations = list(product(*model_info['model_params'].values()))

            for training_days, param_vals_combination in product(model_info['training_days'], param_vals_combinations):
                model_name, model_params = self.get_model_name_and_params(param_names, param_vals_combination, training_days)
                predictions[self.model_name], results[self.model_name] = self.train_and_evaluate(training_days,  model_name, model_params)

            self.initialise_attributes()

        return predictions, results

    def set_model_info(self, model_type, model_info):
        """
        Sets the model information for the current training process.

        Parameters:
        model_type (str): Type of the model (e.g., 'sklearn', 'arima').
        model_info (dict): Dictionary containing model-specific configurations and parameters.
        """
        self.model_type = model_type
        self.model_info = model_info
        self.scale_data_flag = model_info["scale_data"]
        self.model_implementation = model_info["model_implementation"]

    def get_model_name_and_params(self, param_names, param_vals_combination, training_days):
        """
        Constructs the model name and parameters.

        Parameters:
        param_names (list): List of parameter names.
        param_vals_combination (list): List of parameter values corresponding to param_names.
        training_days (int): Number of days for the training period.

        Returns:
        tuple: Contains the model name (str) and model parameters (dict).
        """
        model_name = f"{self.model_type}_" + "_".join(
            f"{name}_{val}" for name, val in zip(param_names, param_vals_combination))
        model_name += f"_train_{training_days}d" if training_days is not None else ""
        model_params = {name: val for name, val in zip(param_names, param_vals_combination)}

        return model_name, model_params

    def train_and_evaluate(self, training_days, model_name, model_params):
        """
        Trains the model and evaluates predictions.

        Parameters:
        training_days (int): Number of days for the training period.
        model_name (str): Name of the model configuration.
        model_params (dict): Parameters for the model configuration.

        Returns:
        tuple: Contains predictions and evaluation results for the given model configuration.
        """
        self.model_name = model_name
        self.model_params = model_params
        predictions = self.get_predictions(training_days)
        results = self.evaluate_predictions(training_days, predictions)
        self.log_results(results, training_days)
        return predictions, results

    def get_predictions(self, training_days):
        """
        Generates predictions for train, validation, and test datasets using a rolling window approach.

        Parameters:
        training_days (int): Number of days for the training period.

        Returns:
        dict: Predictions for each dataset ('train', 'val', 'test').
        """
        predictions = {name: [] for name in self.dataset_names}
        startdate_val_list = pd.date_range(self.startdate_val,
                                           self.startdate_val + timedelta(days=self.validation_days - 1),
                                           freq=f'{self.horizon_days_val}D')

        startdate_test_list = pd.date_range(self.startdate_test,
                                           self.startdate_test + timedelta(days=self.test_days - 1),
                                           freq=f'{self.horizon_days_test}D')

        for i, (current_startdate_val, current_startdate_test) in enumerate(zip(startdate_val_list, startdate_test_list)):
            current_startdate_val = current_startdate_val.date()
            current_startdate_test = current_startdate_test.date()
            is_first_training_window = (i == 0)
            self.filenames = self.create_file_paths(current_startdate_val, current_startdate_test, training_days)
            temp_predictions = self.load_predictions(current_startdate_val, current_startdate_test, training_days, is_first_training_window)

            for dataset_name in self.dataset_names:
                if self.save_predictions_flag[dataset_name]:
                    predictions[dataset_name].append(temp_predictions[dataset_name])

        dataset_names = ['val', 'test'] if self.model_implementation == 'statsmodels' else self.dataset_names

        for dataset_name in dataset_names:
            if self.save_predictions_flag[dataset_name]:
                predictions[dataset_name] = np.concatenate(predictions[dataset_name])

        return predictions

    def create_file_paths(self, startdate_val, startdate_test, training_days):
        """
        Creates file paths for saving predictions and model cache based on training and validation dates.

        Parameters:
        startdate_val (datetime.date): The start date for the validation period.
        startdate_test (datetime.date): The start date for the test period.
        training_days (int): Number of days for the training period.

        Returns:
        dict: A dictionary containing file paths for training, validation, test predictions, and model cache.
        """
        data_dates = self.create_train_val_test_dates(startdate_val, startdate_test, training_days)
        return {
            'pred_train': f".\\predictions\\predictions_train_{self.model_name}_train_period_{data_dates['startdate_train']}_{data_dates['enddate_train']}.npy",
            'pred_val': f".\\predictions\\predictions_val_{self.model_name}_val_period_{data_dates['startdate_val']}_{data_dates['enddate_val']}.npy",
            'pred_test': f".\\predictions\\predictions_test_{self.model_name}_test_period_{data_dates['startdate_test']}_{data_dates['enddate_test']}.npy",
            'model_cache': f".\\models\\model_cache_{self.model_name}_train_period_{data_dates['startdate_train']}_{data_dates['enddate_train']}.joblib"
        }

    def create_train_val_test_dates(self, startdate_val, startdate_test, training_days):
        """
        Calculates the start and end dates for the training, validation, and test periods.

        Parameters:
        startdate_val (datetime.date): The start date for the validation period.
        startdate_test (datetime.date): The start date for the test period.
        training_days (int): Number of days for the training period.

        Returns:
        dict: A dictionary containing the start and end dates for training, validation, and test periods.
        """
        enddate_train = startdate_val - timedelta(days=1)
        startdate_train = enddate_train - timedelta(days=training_days-1)
        enddate_val = startdate_val + timedelta(days=self.horizon_days_val - 1)
        enddate_test = startdate_test + timedelta(days=self.horizon_days_test - 1)
        return {
            'startdate_train': startdate_train,
            'enddate_train': enddate_train,
            'startdate_val': startdate_val,
            'enddate_val': enddate_val,
            'startdate_test': startdate_test,
            'enddate_test': enddate_test,
        }

    def load_predictions(self, startdate_val, startdate_test, training_days, is_first_training_window):
        """
        Loads predictions from files if they exist, otherwise generates predictions using the model.

        Parameters:
        startdate_val (datetime.date): The start date for the validation period.
        startdate_test (datetime.date): The start date for the test period.
        training_days (int): Number of days for the training period.
        is_first_training_window (bool): Flag indicating if this is the first training window.

        Returns:
        dict: Predictions for train, validation, and test datasets.
        """
        predictions = {
            'train': self._load_predictions(self.filenames['pred_train']),
            'val': self._load_predictions(self.filenames['pred_val']),
            'test': self._load_predictions(self.filenames['pred_test']),
        }

        if self.save_predictions_flag['val'] and predictions['val'] is None:
            datasets, temp_covariate_names = self.split_train_val_test_data(startdate_val, startdate_test, training_days)
            model = self.initialize_model(datasets, temp_covariate_names)
            predictions['train'], predictions['val'], _ = self.generate_predictions(model, datasets, is_first_training_window)

        if self.save_predictions_flag['test'] and predictions['test'] is None:
            datasets, temp_covariate_names = self.split_train_val_test_data(startdate_val, startdate_test, training_days)
            model = self.initialize_model(datasets, temp_covariate_names)
            _, _, predictions['test'] = self.generate_predictions(model, datasets, is_first_training_window)

        return predictions

    def _load_predictions(self, filename):
        """
        Loads predictions from a given file if it exists.

        Parameters:
        filename (str): The path to the file containing the predictions.

        Returns:
        np.ndarray or None: The loaded predictions if the file exists, otherwise None.
        """
        try:
            if os.path.exists(filename):
                print(f"Loading predictions from {filename}")
                return np.load(filename)
        except Exception as e:
            print(f"Error loading predictions from {filename}: {e}")
        return None

    def split_train_val_test_data(self, startdate_val, startdate_test, training_days):
        """
        Splits the dataframe into training, validation, and test sets and scales the data if required.

        Parameters:
        startdate_val (datetime.date): The start date for the validation period.
        startdate_test (datetime.date): The start date for the test period.
        training_days (int): Number of days for the training period.

        Returns:
        tuple: A dictionary of datasets and a list of covariate names.
        """
        try:
            data_dates = self.create_train_val_test_dates(startdate_val, startdate_test, training_days)
            df_subset = self.subset_data(data_dates)
            covariate_names, categorical_covariate_names = self.get_covariate_names(df_subset)
            datasets = self.create_datasets(df_subset, data_dates, covariate_names)
            numerical_names, numerical_inds = self.get_numerical_covariate_names_and_indices(covariate_names, categorical_covariate_names)

            if self.scale_data_flag:
                datasets = self.scale_data(datasets, numerical_inds)

            return datasets, covariate_names
        except Exception as e:
            print(f"Error in split_train_val_test_data: {e}")
            raise

    def subset_data(self, data_dates):
        """
        Subsets the dataframe to the training and testing date range and generates dummy features for categorical variables.

        Parameters:
        data_dates (dict): A dictionary containing the start and end dates for training and testing periods.

        Returns:
        pd.DataFrame: A subset of the dataframe with dummy features for categorical variables.
        """
        df_subset = self.df[(self.df['datetime'].dt.date >= data_dates['startdate_train']) & (
                self.df['datetime'].dt.date <= data_dates['enddate_test'])].copy()
        if self.model_info['generate_dummy_features']:
            df_subset = self._generate_dummy_features(df_subset)
        return df_subset

    def _generate_dummy_features(self, df):
        """
        Generates dummy features for categorical variables in the dataframe.

        Parameters:
        df (pd.DataFrame): The input dataframe.

        Returns:
        pd.DataFrame: The dataframe with dummy features for categorical variables.
        """

        df_column_names = df.columns.to_list()
        if self.categorical_covariate_names:
            df = pd.get_dummies(df, columns=self.categorical_covariate_names, dtype=float, drop_first=True)
            self.dummy_feature_names = list(set(df) - set(df_column_names) - set(self.categorical_covariate_names))

        return df

    def get_covariate_names(self, df_subset):
        """
        Retrieves covariate names excluding datetime, target, and categorical covariate names.

        Parameters:
        df_subset (pd.DataFrame): The subset of the dataframe.

        Returns:
        list: A list of covariate names.
        """

        covariate_names = [c for c in df_subset.columns if c in (self.covariate_names + self.dummy_feature_names)]

        if self.model_info['generate_dummy_features']:
            covariate_names = [c for c in covariate_names if c not in self.categorical_covariate_names]
            categorical_covariate_names = self.dummy_feature_names
        else:
            categorical_covariate_names = self.categorical_covariate_names

        return covariate_names, categorical_covariate_names

    def create_datasets(self, df, data_dates, covariate_names):
        """
        Creates training, validation, and test datasets from the subset dataframe.

        Parameters:
        df (pd.DataFrame): dataframe.
        data_dates (dict): A dictionary containing the start and end dates for training, validation, and test periods.
        covariate_names (list): A list of covariate names.

        Returns:
        dict: A dictionary containing the datasets.
        """
        datasets = {}
        for dataset_name in self.dataset_names:
            _ind = (df['datetime'].dt.date >= data_dates[f'startdate_{dataset_name}']) & (
                        df['datetime'].dt.date <= data_dates[f'enddate_{dataset_name}'])
            datasets[f'X_{dataset_name}'] = df[_ind][covariate_names].values
            datasets[f'y_{dataset_name}'] = df[_ind][self.target_name].values
        return datasets

    def scale_data(self, datasets, numerical_inds):
        """
        Scales the numerical features in the datasets using StandardScaler.

        Parameters:
        datasets (dict): The dictionary containing datasets to be scaled.
        numerical_inds (list): List of indices for numerical covariates.

        Returns:
        dict: The scaled datasets.
        """
        model_cache = self.load_model_cache(self.filenames['model_cache'])
        covariate_scaler = StandardScaler()
        covariate_scaler.fit(datasets['X_train'][:, numerical_inds])
        model_cache['covariate_scaler'] = covariate_scaler

        target_scaler = StandardScaler()
        target_scaler.fit(datasets['y_train'].reshape(-1, 1))
        model_cache['target_scaler'] = target_scaler

        self.save_model_cache(self.filenames['model_cache'], model_cache)

        datasets['X_train'][:, numerical_inds] = model_cache['covariate_scaler'].transform(
            datasets['X_train'][:, numerical_inds])
        datasets['X_val'][:, numerical_inds] = model_cache['covariate_scaler'].transform(
            datasets['X_val'][:, numerical_inds])
        datasets['X_test'][:, numerical_inds] = model_cache['covariate_scaler'].transform(
            datasets['X_test'][:, numerical_inds])
        datasets['y_train'] = model_cache['target_scaler'].transform(datasets['y_train'].reshape(-1, 1)).reshape(-1)

        return datasets

    def get_numerical_covariate_names_and_indices(self, covariate_names, categorical_covariate_names):
        """
        Identifies numerical covariate names and their indices.

        Parameters:
        covariate_names (list): List of covariate names.

        Returns:
        tuple: A tuple containing a list of numerical covariate names and their indices.
        """
        numerical_names = set(covariate_names)

        if categorical_covariate_names:
            numerical_names -= set(categorical_covariate_names)

        if self.binary_covariate_names:
            numerical_names -= set(self.binary_covariate_names)

        numerical_names = list(numerical_names)
        numerical_indices = [i for i, name in enumerate(covariate_names) if name in numerical_names]

        return numerical_names, numerical_indices

    def initialize_model(self, datasets, covariate_names):
        """
        Default implementation of model initialisation logic.
        Can be overridden by subclasses if needed

        Parameters:
        datasets: A dictionary of datasets.
        covariate_names: A list of covariate names
        """
        pass

    def generate_predictions(self, model, datasets, is_first_training_window):
        """
        Generates predictions for training, validation, and test datasets using the provided model.

        Parameters:
        model: The model to be trained and used for generating predictions.
        datasets (dict): A dictionary containing training, validation, and test datasets.
        is_first_training_window (bool): Flag indicating if this is the first training window.

        Returns:
        tuple: Predictions for training, validation, and test datasets.
        """

        pred_train = np.array([])

        if self.model_implementation != 'statsmodels':
            model.fit(datasets['X_train'], datasets['y_train'], **self.model_info['fit_params'])

            train_data = datasets['X_train'] if is_first_training_window else datasets['X_train'][-24:]
            pred_train = self.save_predictions(model, train_data, None,self.filenames['pred_train'],
                                               self.filenames['model_cache']) if self.save_predictions_flag['train'] else np.array([])

        pred_val = self.save_predictions(model, datasets['X_val'], self.horizon_days_val, self.filenames['pred_val'],
                                         self.filenames['model_cache']) if self.save_predictions_flag['val'] else np.array([])
        pred_test = self.save_predictions(model, datasets['X_test'], self.horizon_days_test, self.filenames['pred_test'],
                                          self.filenames['model_cache']) if self.save_predictions_flag['test'] else np.array([])

        return pred_train, pred_val, pred_test

    def save_predictions(self, model, data, horizon_days, predictions_filename, model_cache_filename):
        """
        Generates and saves predictions using the provided model.

        Parameters:
        model: The model used to generate predictions.
        data (np.ndarray): The input data for prediction.
        predictions_filename (str): The filename where predictions will be saved.
        model_cache_filename (str, optional): The filename of the model cache for scaling predictions.

        Returns:
        np.ndarray: The generated predictions.
        """
        try:
            if self.model_implementation == "sklearn":
                y_pred = model.predict(data)
                if self.scale_data_flag and model_cache_filename:
                    model_cache = self.load_model_cache(model_cache_filename)
                    y_pred = model_cache['target_scaler'].inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
            elif self.model_implementation == 'statsmodels':
                y_pred = model.forecast(horizon_days * 24).values

            print(f"Saving {predictions_filename}")
            with open(predictions_filename, 'wb') as f:
                np.save(f, y_pred)

            return y_pred
        except Exception as e:
            print(f"Error saving predictions to {predictions_filename}: {e}")
            return None

    def load_model_cache(self, filename):
        """
        Loads a model cache from a specified file if it exists.

        Parameters:
        filename (str): The path to the file containing the model cache.

        Returns:
        dict: The loaded model cache if the file exists and is loaded successfully, otherwise an empty dictionary.
        """
        try:
            if os.path.exists(filename):
                print(f"Loading model cache from {filename}")
                with open(filename, 'rb') as f:
                    return joblib.load(f)
        except Exception as e:
            print(f"Error loading model cache from {filename}: {e}")
        return {}

    def save_model_cache(self, filename, model_cache):
        """
        Saves the model cache to a specified file.

        Parameters:
        filename (str): The path to the file where the model cache will be saved.
        model_cache (dict): The model cache to be saved.

        Returns:
        None
        """
        try:
            print(f"Saving model cache to {filename}")
            with open(filename, 'wb') as f:
                joblib.dump(model_cache, f)
        except Exception as e:
            print(f"Error saving model cache to {filename}: {e}")

    def evaluate_predictions(self, training_days, predictions):
        """
        Evaluates predictions for train, validation, and test datasets.

        Parameters:
        training_days (int): Number of days for the training period.
        predictions (dict): Predicted values for each dataset ('train', 'val', 'test').

        Returns:
        dict: Evaluation metrics for each dataset.
        """

        if self.model_type == 'naive':
            datasets, _ = self.split_train_val_test_data(self.startdate_val, self.startdate_test, training_days)
            y_train = datasets[f'y_train']
        else:
            y_train = self.generate_y_train_rolling(training_days)

        dataset_names = ['val', 'test'] if self.model_implementation == 'statsmodels' else self.dataset_names
        dataset_targets = [self.y_val, self.y_test] if self.model_implementation == 'statsmodels' else \
            [y_train, self.y_val, self.y_test]

        results = {}

        for dataset_name, dataset_target in zip(dataset_names, dataset_targets):
            if self.save_predictions_flag[dataset_name]:
                results[dataset_name] = self._evaluate_predictions(dataset_name, dataset_target,
                                                                   predictions[dataset_name])
            else:
                results[dataset_name] = None

        return results

    def generate_y_train_rolling(self, training_days):
        """
        Generates the rolling training target values for a specified training period.

        Parameters:
        training_days (int): Number of days for the training period.

        Returns:
        np.ndarray: The target values for the rolling training period.
        """
        train_startdate = self.startdate_val - timedelta(days=training_days)
        train_enddate = self.startdate_val + timedelta(days=self.validation_days - 2)
        train_indices = (self.df['datetime'].dt.date >= train_startdate) & (self.df['datetime'].dt.date <= train_enddate)
        y_train = self.df.loc[train_indices, self.target_name].values

        return y_train

    def _evaluate_predictions(self, dataset_name, y_true, y_pred):
        """
        Evaluates the model predictions against the true values using various metrics.

        Parameters:
        dataset_name (str): The name of the dataset being evaluated ('train', 'val', or 'test').
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values by the model.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        mae = mean_absolute_error(y_true, y_pred)
        res = {
            "dataset": dataset_name,
            "model": self.model_name,
            "rmse": np.sqrt(mae),
            "r2": r2_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
            "smape": self.symmetric_mean_absolute_percentage_error(y_true, y_pred),
            "mae": mae
        }
        return res

    def symmetric_mean_absolute_percentage_error(self, y_true, y_pred):
        """
        Calculates the symmetric mean absolute percentage error (SMAPE) between the true and predicted values.

        Parameters:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values by the model.

        Returns:
        float: The SMAPE value as a percentage.
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        smape = np.nanmean(np.abs(y_pred - y_true) / np.where(denominator == 0, np.nan, denominator)) * 100
        return smape

    def log_results(self, results, training_days):
        """
        Logs the model parameters and evaluation metrics to MLflow.

        Parameters:
        results (dict): Dictionary containing evaluation metrics for each dataset.
        training_days (int): Number of days for the training period.
        """
        with mlflow.start_run(run_name=self.model_name):
            params = self.model_params.copy()
            params["training_days"] = training_days
            mlflow.log_params(params)

            for dataset_name, metrics in results.items():
                if metrics:
                    for metric_name, value in metrics.items():
                        if metric_name not in ["dataset", "model"]:
                            mlflow.log_metric(f"{dataset_name}_{metric_name}", value)

class RollingWindowSklearn(RollingWindowBase):
    """
    Implements rolling window training and evaluation for sklearn models.

    Inherits from RollingWindowBase to provide additional functionality specific to sklearn models.
    """

    def initialize_model(self, datasets, covariate_names):
        """
        Initializes and returns a sklearn model instance with specified parameters.

        Parameters:
        datasets: A dictionary of datasets.
        covariate_names: A list of covariate names

        Returns:
        model: The initialized sklearn model instance.
        """
        model = deepcopy(self.model_info['model_instance'])
        model.set_params(**self.model_params)
        return model

class RollingWindowLightGBM(RollingWindowBase):
    """
    Implements rolling window training and evaluation for sklearn models.

    Inherits from RollingWindowBase to provide additional functionality specific to sklearn models.
    """

    def initialize_model(self, datasets, covariate_names):
        """
        Initializes and returns a sklearn model instance with specified parameters.

        Returns:
        model: The initialized sklearn model instance.
        """
        self.get_fit_params(datasets, covariate_names)
        model = deepcopy(self.model_info['model_instance'])
        if self.model_params:
            model.set_params(**self.model_params)
        return model

    def get_fit_params(self, datasets, covariate_names):
        """
        Sets the parameters for fitting the LightGBM model

        Parameters:
        kwargs (dict): Dictionary containing parameter
        """
        ind_categorical_covariate_names = [
            i for i, name in enumerate(covariate_names) if name in self.categorical_covariate_names
        ]

        self.model_info['fit_params'] = {
            'categorical_feature': ind_categorical_covariate_names,
            'eval_set': [(datasets['X_val'], datasets['y_val'])]
        }



def statsmodel_fit_summary(model, plot_title):
    print("#" * 20)
    print("#" * 20)
    print(f"Fitting {plot_title}")
    print("")
    # summary of fit model
    print(model.summary())
    # Residual Plot
    residuals = pd.DataFrame(model.resid)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    residuals.plot(ax=axs[0], title="Residuals vs Time")
    residuals.plot(kind='kde', ax=axs[1], title="Residuals distibution")
    fig.suptitle(plot_title)
    plt.show()
    print("#" * 20)
    print("#" * 20)

class RollingWindowSarimax(RollingWindowBase):

    def split_train_val_test_data(self, startdate_val, startdate_test, training_days):
        """
        Splits the dataframe into training, validation, and test sets and scales the data if required.

        Parameters:
        startdate_val (datetime.date): The start date for the validation period.
        startdate_test (datetime.date): The start date for the test period.
        training_days (int): Number of days for the training period.

        Returns:
        tuple: A dictionary of datasets and a list of covariate names.
        """
        try:
            data_dates = self.create_train_val_test_dates(startdate_val, startdate_test, training_days)
            df_subset = self.subset_data(data_dates)
            datasets = self.create_datasets(df_subset, data_dates)

            return datasets, []
        except Exception as e:
            print(f"Error in split_train_val_test_data: {e}")
            raise

    def create_datasets(self, df_subset, data_dates):
        """
        Creates training, validation, and test datasets from the subset dataframe.

        Parameters:
        df_subset (pd.DataFrame): The subset of the dataframe.
        data_dates (dict): A dictionary containing the start and end dates for training, validation, and test periods.
        covariate_names (list): A list of covariate names.

        Returns:
        dict: A dictionary containing the datasets.
        """
        datasets = {}
        for dataset_name in self.dataset_names:
            _ind = (df_subset['datetime'].dt.date >= data_dates[f'startdate_{dataset_name}']) & (
                        df_subset['datetime'].dt.date <= data_dates[f'enddate_{dataset_name}'])
            datasets[f'X_{dataset_name}'] = df_subset[_ind].set_index('datetime')[self.target_name]
            datasets[f'X_{dataset_name}'].index.freq = 'h'
        return datasets

    def initialize_model(self, datasets, covariate_names):
        """
        Initializes and returns a sklearn model instance with specified parameters.

        Returns:
        model: The initialized sklearn model instance.
        """
        model = SARIMAX(datasets['X_train'], order=self.model_params['order'],
                        seasonal_order=self.model_params['seasonal_order']).fit()
        return model

class RollingWindowNaive(RollingWindowBase):

    def split_train_val_test_data(self, startdate_val, startdate_test, training_days):
        """
        Splits the dataframe into training, validation, and test sets and scales the data if required.

        Parameters:
        startdate_val (datetime.date): The start date for the validation period.
        training_days (int): Number of days for the training period.

        Returns:
        tuple: A dictionary of datasets and a list of covariate names.
        """
        try:
            data_dates = self.create_train_val_test_dates(startdate_val, startdate_test, training_days)
            datasets = self.create_datasets(self.df.copy(), data_dates)

            return datasets, []
        except Exception as e:
            print(f"Error in split_train_val_test_data: {e}")
            raise

    def create_train_val_test_dates(self, startdate_val, startdate_test, training_days):
        """
        Calculates the start and end dates for the training, validation, and test periods.

        Parameters:
        startdate_val (datetime.date): The start date for the validation period.
        startdate_test (datetime.date): The start date for the test period.
        training_days (int): Number of days for the training period.

        Returns:
        dict: A dictionary containing the start and end dates for training, validation, and test periods.
        """
        enddate_train = startdate_val - timedelta(days=1)
        startdate_train = enddate_train - timedelta(days=training_days-self.model_params['lookback_horizon_days'] - 1)
        enddate_val = startdate_val + timedelta(days=self.horizon_days_val - 1)
        enddate_test = startdate_test + timedelta(days=self.horizon_days_test - 1)
        return {
            'startdate_train': startdate_train,
            'enddate_train': enddate_train,
            'startdate_val': startdate_val,
            'enddate_val': enddate_val,
            'startdate_test': startdate_test,
            'enddate_test': enddate_test,
        }

    def create_datasets(self, df, data_dates):
        """
        Creates training, validation, and test datasets from the subset dataframe.

        Parameters:
        df_subset (pd.DataFrame): The subset of the dataframe.
        data_dates (dict): A dictionary containing the start and end dates for training, validation, and test periods.

        Returns:
        dict: A dictionary containing the datasets.
        """
        datasets = {}

        for dataset_name in self.dataset_names:
            temp_startdate_covariate = data_dates[f'startdate_{dataset_name}'] - timedelta(days=self.model_params['lookback_horizon_days'])

            covariate_ind = (df["datetime"].dt.date >= temp_startdate_covariate) & (
                    df["datetime"].dt.date <= data_dates[f'enddate_{dataset_name}'])
            target_ind = (df["datetime"].dt.date >= data_dates[f'startdate_{dataset_name}']) & (
                    df["datetime"].dt.date <= data_dates[f'enddate_{dataset_name}'])

            datasets[f'X_{dataset_name}'] = df[covariate_ind][self.target_name].values
            datasets[f'y_{dataset_name}'] = df[target_ind][self.target_name].values

            del temp_startdate_covariate
        return datasets

    def get_predictions(self, training_days):
        """
        Generates predictions for train, validation, and test datasets using a rolling window approach.

        Parameters:
        training_days (int): Number of days for the training period.

        Returns:
        dict: Predictions for each dataset ('train', 'val', 'test').
        """

        self.filenames = self.create_file_paths(self.startdate_val, self.startdate_test, training_days)
        predictions = self.load_predictions(self.startdate_val, self.startdate_test, training_days, is_first_training_window=True)

        return predictions

    def initialize_model(self,datasets, covariate_names):
        """
        Initializes and returns a naive model instance with specified parameters.

        Returns:
        model: The initialized sklearn model instance.
        """
        return Naive(lookback_horizon_days=self.model_params['lookback_horizon_days'])