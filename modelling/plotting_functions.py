import os
import joblib
from lightgbm import LGBMRegressor
from copy import deepcopy
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from IPython.display import HTML, display

class PlotLightGBMTrainingPredictions:

    def __init__(self, df_train, covariate_names, categorical_covariate_names, target_name, model_name):
        """
        Initializes the PlotLightGBMTrainingPredictions class.

        Parameters:
        df_train (pd.DataFrame): Training data.
        covariate_names (List[str]): List of covariate names.
        categorical_covariate_names (List[str]): List of categorical covariate names.
        target_name (str): Target variable name.
        model_name (str): Name of the model.
        """
        self.df_train = df_train
        self.covariate_names = covariate_names
        self.target_name = target_name
        self.model_name = model_name
        self.categorical_covariate_names = categorical_covariate_names

    def generate_plot(self, training_days):
        """
        Generates an interactive plot of the LightGBM predictions on the training data.

        Parameters:
        training_days (int): Number of training days.
        """
        df_plot_data = self.generate_plot_data(training_days)
        self._generate_plot(df_plot_data, training_days)

    def generate_plot_data(self, training_days):
        """
        Generates data for the plot.

        Parameters:
        training_days (int): Number of training days.

        Returns:
        pd.DataFrame: DataFrame containing datetime, ground truth, and predictions.
        """
        covariate_names = self.get_covariate_names(training_days)
        ind_categorical_features = self.get_ind_categorical_features(covariate_names)
        datasets = self.split_data(training_days, covariate_names)
        model_params = self.load_model_cache(training_days)
        if 'num_iterations' in model_params:
            model_params['n_estimators'] = model_params.pop('num_iterations')
        model = self.fit_model(model_params, datasets, ind_categorical_features)
        predictions = model.predict(datasets['X_train'])
        df_plot_data = self.concatenate_plot_data(datasets['datetime_train'], datasets['y_train'], predictions)

        return df_plot_data

    def get_covariate_names(self, training_days):
        """
        Gets the model covariate names.

        Parameters:
        training_days (int): Number of training days.

        Returns:
        List[str]: List of covariate names.
        """
        covariate_names = deepcopy(self.covariate_names)
        if training_days < 365:
            covariate_names.remove('target_month')
        return covariate_names

    def get_ind_categorical_features(self, covariate_names):
        """
        Gets the indices of the categorical features.

        Parameters:
        covariate_names (List[str]): List of covariate names.

        Returns:
        List[int]: List of indices of categorical features.
        """

        return [i for i, feature in enumerate(covariate_names) if feature in self.categorical_covariate_names]

    def split_data(self, training_days, covariate_names):
        """
        Splits data into training and validation sets.

        Parameters:
        training_days (int): Number of training days.
        covariate_names (List[str]): List of covariate names.

        Returns:
        Dict[str, Any]: Dictionary containing training and validation data.
        """
        training_hours = min(training_days * 24, 5000)

        return {
            'datetime_train': self.df_train['datetime'][:training_hours],
            'X_train': self.df_train[covariate_names].values[:training_hours],
            'y_train': self.df_train[self.target_name].values[:training_hours],
            'X_val': self.df_train[covariate_names].values[training_hours:training_hours + 24],
            'y_val': self.df_train[self.target_name].values[training_hours:training_hours + 24],
        }

    def load_model_cache(self, training_days):
        """
        Loads a model cache.

        Parameters:
        training_days (int): Number of training days.

        Returns:
        Dict[str, Any]: Model parameters.
        """
        filename = self.get_filename(training_days)
        try:
            if os.path.exists(filename):
                print(f"Loading model cache from {filename}")
                with open(filename, 'rb') as f:
                    return joblib.load(f)
        except Exception as e:
            print(f"Error loading model cache from {filename}: {e}")
        return {}

    def get_filename(self, training_days):
        """
        Creates the model cache filename.

        Parameters:
        training_days (int): Number of training days.

        Returns:
        str: Model cache filename.
        """
        return f"..\\models\\hyperparams_{self.model_name}_{training_days}d.joblib"

    def fit_model(self, params, datasets, ind_categorical_features):
        """
        Fits the model.

        Parameters:
        params (Dict[str, Any]): Model parameters.
        datasets (Dict[str, Any]): Dictionary containing training and validation data.
        ind_categorical_features (List[int]): List of indices of categorical features.

        Returns:
        LGBMRegressor: Trained LightGBM model.
        """
        model = LGBMRegressor(n_jobs=-1, random_state=0, verbosity=-1, **params)
        model.fit(datasets['X_train'], datasets['y_train'], categorical_feature=ind_categorical_features,
                  eval_set=[(datasets['X_val'], datasets['y_val'])])
        return model

    def concatenate_plot_data(self, datetime, ground_truth, predictions):
        """
        Concatenates datetime, ground_truth, and predictions into a DataFrame.

        Parameters:
        datetime (pd.Series): Datetime series.
        ground_truth (np.ndarray): Ground truth values.
        predictions (np.ndarray): Model predictions.

        Returns:
        pd.DataFrame: DataFrame containing datetime, ground truth, and predictions.
        """
        data = np.concatenate([ground_truth.reshape(-1, 1), predictions.reshape(-1, 1)], axis=1)
        df = pd.DataFrame(data=data, columns=['ground_truth', 'prediction'])
        df['datetime'] = datetime
        return df

    def _generate_plot(self, df_plot_data, training_days):
        """
        Generates an interactive Altair plot.

        Parameters:
        df_plot_data (pd.DataFrame): DataFrame containing plot data.
        training_days (int): Number of training days.
        """
        xaxis_config = alt.Axis(
            grid=False,
            labelFontSize=14,  # Increase font size for labels
            titleFontSize=16,  # Increase font size for axis title
            titleX=0,  # Align title to the left
            titleAlign='left',
            titlePadding=10  # Add some padding between axis and title
        )

        yaxis_config = alt.Axis(
            grid=False,
            labelFontSize=14,  # Increase font size for labels
            titleFontSize=16,  # Increase font size for axis title
        )

        predictions = alt.Chart(df_plot_data).mark_line(color='orange').encode(
            x=alt.X('datetime:T', title='Datetime', axis=xaxis_config),
            y=alt.Y('prediction:Q', title='Price (€/MWh)', axis=yaxis_config)
        ).properties(
            width=10000,
            height=500
        )

        ground_truth = alt.Chart(df_plot_data).mark_point(color='blue', size=30, opacity=0.6).encode(
            x=alt.X('datetime:T', title='Datetime', axis=xaxis_config),
            y=alt.Y('ground_truth:Q', title='Price (€/MWh)', axis=yaxis_config)
        ).properties(
            width=10000,
            height=500
        )

        chart = alt.layer(
            predictions,
            ground_truth
        ).properties(
            title=alt.TitleParams(
                text=f'Training Predictions - {self.model_name}_{training_days}d',
                anchor='start',  # Align title to the left
                fontSize=18  # Increase title font size
            )
        ).interactive()

        chart.display()

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


def plot_predictions_interactive(df, dataset_name):
    # Melt the dataframe to have a long format suitable for Altair
    dfm = df.melt(ignore_index=False).reset_index()
    dfm.columns = ['datetime', 'model', 'price']
    df_pred_gt = dfm[dfm['model'] == 'ground_truth'].copy()
    dfm = dfm[dfm['model'] != 'ground_truth'].copy()

    # Inject Custom CSS to add space between radio button options
    display(HTML("""
    <style>
        .vega-bindings label {
            margin-right: 10px;  /* Adjust this value to increase the spacing */
        }
    </style>
    """))

    # Define the model names excluding ground_truth
    model_names = ['sarimax', 'lasso', 'ridge', 'elasticnet', 'linear_gam', 'light_gbm']

    # Create radio button input for model selection
    input_dropdown = alt.binding_radio(
        options=model_names + [None],
        labels=[name + ' ' for name in model_names] + ['All'],
        name='Model: '
    )
    selection = alt.selection_point(
        fields=['model'],
        bind=input_dropdown,
    )

    # Configuration for axes
    xaxis_config = alt.Axis(
        grid=False,
        labelFontSize=14,
        titleFontSize=16,
        titleX=0,
        titleAlign='left',
        titlePadding=10,
        format="%d %b %y"
    )

    yaxis_config = alt.Axis(
        grid=False,
        labelFontSize=14,
        titleFontSize=16
    )

    # Chart for ground truth values
    ground_truth = alt.Chart(df_pred_gt).mark_point(color='blue', size=30, opacity=0.4).encode(
        x=alt.X('datetime:T', title='Datetime', axis=xaxis_config),
        y=alt.Y('price:Q', title='Price (€/MWh)', axis=yaxis_config),
        tooltip=['datetime:T', 'price:Q', 'model:N']
    )

    # Chart for model predictions
    predictions = alt.Chart(dfm).mark_line().encode(
        x=alt.X('datetime:T', title='Datetime', axis=xaxis_config),
        y=alt.Y('price:Q', title='Price (€/MWh)', axis=yaxis_config),
        color=alt.condition(
            selection,
            alt.Color('model:N', legend=alt.Legend(title="Model"),
                      scale=alt.Scale(domain=['ground_truth'] + model_names)),
            alt.value('grey')
        ),
        opacity=alt.condition(
            selection,
            alt.value(1),
            alt.value(0)
        ),
        tooltip=['datetime:T', 'price:Q', 'model:N']
    ).add_params(
        selection
    )

    # Combine the charts
    chart = alt.layer(
        ground_truth,
        predictions
    ).properties(
        title=alt.TitleParams(
            text=f'{dataset_name} Predictions',
            anchor='start',
            fontSize=18
        ),
        width=15000,
        height=500
    ).interactive()

    return chart

def plot_cumulative_absolute_error(df, dataset):

    model_names = [
        'sarimax',
        'lasso',
        'ridge',
        'elasticnet',
        'linear_gam',
        'light_gbm'
    ]

    df_ae = df['datetime'].copy().to_frame()

    for model in model_names:
        df_ae[model] = np.abs(df['ground_truth'].sub(df[model]))

    df_ae = df_ae.set_index('datetime')

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 7))
    df_ae.cumsum().plot(ax=ax)
    ax.legend(fontsize=12)

    # Set font size for axis titles
    ax.set_title(f'Cumulative Absolute {dataset} Error', fontsize=18)  # Increase the font size of the title
    ax.set_xlabel('Date', fontsize=12)  # Increase the font size of the x-axis title
    ax.set_ylabel('Cumulative Absolute Error', fontsize=12)  # Increase the font size of the y-axis title

    # Change the font size and style of the x and y ticks
    ax.tick_params(axis='x', labelsize=12)  # Adjust fontsize and rotation for x ticks
    ax.tick_params(axis='y', labelsize=12)  # Adjust fontsize for y ticks

    plt.show()