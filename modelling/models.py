

class Naive:
    """
    Naive forecasting model that predicts the future values based on historical data.
    """
    def __init__(self, lookback_horizon_days):
        """
        Initializes the Naive model with the given lookback horizon.

        Parameters:
        lookback_horizon_days (int): The number of days to look back for making predictions.
        """
        self.lookback_horizon_days = lookback_horizon_days
        self.lookback_horizon_hours = lookback_horizon_days * 24

    def fit(self, X, y, **kwargs):
        """
        This method is a placeholder to maintain consistency with other models that require fitting.
        """
        pass

    def predict(self, y_true):
        """
        Makes predictions using the naive forecasting method.

        Parameters:
        y_true (array-like): The true values of the time series.

        Returns:
        array-like: The predicted values.
        """
        if len(y_true) <= self.lookback_horizon_hours:
            raise ValueError("Input data is shorter than the lookback horizon.")

        y_pred = y_true[:-self.lookback_horizon_hours]
        return y_pred