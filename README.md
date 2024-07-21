# Netherlands Electricity Price Prediction

## Executive Summary

### Project Motivation

The sharp increase in European energy prices in 2022, which saw energy and supply costs surge by 110% compared to 2020, highlighted the urgent need for accurate energy price forecasts to help households optimize their energy usage and mitigate financial strain. This project aims to develop a model that generates precise day-ahead energy price forecasts, particularly for the Netherlands, and presents these forecasts through a user-friendly dashboard application.

### Data and Features

Data was sourced from the ENTSO-E transparency platform and OpenWeatherMap, covering energy prices, total load, generation capacity, cross-border flows, and weather variables such as temperature, wind speed, and humidity. Additionally, Dutch public holidays were included as calendar features. Comprehensive data cleaning and feature engineering steps were undertaken to prepare the dataset for modeling.

### Methodology

Various models were explored for forecasting day-ahead electricity prices, including:

1. **Naive Forecasts**: Using prices from 48 hours and one week prior.
2. **Seasonal ARIMA Models**: Incorporating seasonality and autocorrelation factors.
3. **Regularised Linear Models**: Lasso, Ridge, and Elastic Net to handle multicollinearity and feature selection.
4. **Generalised Additive Models (GAMs)**: Capturing smooth, non-linear relationships between features and the target variable.
5. **Gradient Boosting Decision Trees (GBDTs)**: LightGBM for efficient and scalable modeling.

### Results

- **Naive Models**: Demonstrated strong seasonality in the training and validation data but performed poorly on the test data due to volatility.
- **SARIMAX**: Outperformed naive models on validation data but failed to generalize well to the test data, indicating the need for exogenous variables.
- **Regularised Linear Models**: Improved on naive models, with Ridge achieving the lowest test MAE of 22.91 and an \( R^2 \) of 0.41.
- **GAM**: Showed modest improvement over naive models but required large smoothing parameters, resulting in linear-like relationships.
- **LightGBM**: Achieved the best test results among all models, though the performance difference between validation and test data highlighted challenges in handling data volatility and seasonality.

### System Design

The project employed Streamlit for the frontend interface and FastAPI for the backend, facilitating user interaction and efficient processing of model predictions. Docker was used to containerize both frontend and backend, ensuring consistency and ease of deployment. The prototype application was deployed on DigitalOcean.

### Conclusion and Future Work

The developed model successfully generated precise day-ahead electricity price forecasts for the Netherlands, outperforming benchmarks in the literature. Future work will focus on incorporating additional covariates, such as fuel prices, and exploring sophisticated techniques to handle temporal variability in seasonality and volatility to further enhance predictive accuracy.

### Demonstration

<div align="center">
    <img src="images/demo.gif" alt=" " width="800"/>
    <p><strong>Figure 17:</strong> Demonstration of the Application's User Experience.</p>
</div>

### References

- Alvarez, C.F. and Molnar, G., 2021. What is behind soaring energy prices and what happens next?.
- Basanisi, L. (2020, June). Energy consumption of the Netherlands. Retrieved March 29, 2024 from https://www.kaggle.com/datasets/lucabasa/dutch-energy.
- Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. TProc. of the 30th International Conference on Machine Learning (ICML 2013), June 2013, pp. I-115 to I-23.
- Bolton, R., 2022. Natural gas, rising prices and the electricity market.
- ENTSO-E. Detailed Data Descriptions, Version 3, Release 3; 2022. Available at: ENTSO-E Detailed Data Descriptions (Accessed: March 25, 2024).
- ENTSO-E transparency platform. Available at: ENTSO-E Transparency (Accessed: March 25, 2024).
- Friedman, J.H., 2001. Greedy function approximation: a gradient boosting machine. Annals of statistics, pp.1189-1232.
- Hastie, T. & Tibshirani, R. (1986), 'Generalized Additive Models', Statist. Sci. 1 (3), 297--310.
- holidays 0.50. Available at: https://pypi.org/project/holidays/ (Accessed: March 29, 2024).
- Keles, D., Scelle, J., Paraschiv, F. and Fichtner, W., 2016. Extended forecast methods for day-ahead electricity spot prices applying artificial neural networks. Applied energy, 162, pp.218-230.
- Lago, J., De Ridder, F. and De Schutter, B., 2018. Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms. Applied Energy, 221, pp.386-405.
- Lago, J., De Ridder, F., Vrancx, P. and De Schutter, B., 2018. Forecasting day-ahead electricity prices in Europe: The importance of considering market integration. Applied energy, 211, pp.890-903.
- Lago, J., Marcjasz, G., De Schutter, B. and Weron, R., 2021. Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. Applied Energy, 293, p.116983
- Larsen, K. (2015) Gam: The predictive modeling silver bullet, GAM: The Predictive Modeling Silver Bullet. Available at: https://multithreaded.stitchfix.com/blog/2015/07/30/gam/ (Accessed: 10 June 2024). 
- OpenWeather. Available at: https://openweathermap.org/ (Accessed: March 29, 2024).
- pmdarima 2.0.4. Available at: https://pypi.org/project/pmdarima/ (Accessed: March 29, 2024).
- Power Engineering International (2023) Record low European power demand in Q2 as renewables output hits new high, Power Engineering International. Available at: https://www.powerengineeringint.com/world-regions/europe/record-low-european-power-demand-in-q2-as-renewables-output-hits-new-high/ (Accessed: 03 June 2024). 
- Ramírez, S. FastAPI [Computer software]. https://github.com/tiangolo/fastapi
- Servén D., Brummitt C. (2018). pyGAM: Generalized Additive Models in Python. Zenodo. DOI: 10.5281/zenodo.1208723
- Shi Y, Ke G, Soukhavong D, Lamb J, Meng Q, Finley T, Wang T, Chen W, Ma W, Ye Q, Liu T, Titov N, Cortes D (2024). lightgbm: Light Gradient Boosting Machine. R package version 4.3.0.99, https://github.com/Microsoft/LightGBM.
- Smal, T. and Wieprow, J., 2023. Energy security in the context of global energy crisis: economic and financial conditions. Energies, 16(4), p.1605.
- Tertre, M.G., Martinez, I. and Rábago, M.R., 2023. Reasons behind the 2022 energy price increases and prospects for next year.
