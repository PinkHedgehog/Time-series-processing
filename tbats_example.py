from tbats import TBATS
import numpy as np
# required on windows for multi-processing,
# see https://docs.python.org/2/library/multiprocessing.html#windows
if __name__ == '__main__':
    np.random.seed(2342)
    t = np.array(range(0, 160))
    y = 5 * np.sin(t * 2 * np.pi / 7) + 2 * np.cos(t * 2 * np.pi / 30.5) + \
        ((t / 20) ** 1.5 + np.random.normal(size=160) * t / 50) + 10
    
    # Create estimator
    estimator = TBATS(seasonal_periods=[14, 30.5])
    
    # Fit model
    fitted_model = estimator.fit(y)
    
    # Forecast 14 steps ahead
    y_forecasted = fitted_model.forecast(steps=14)
    
    # Summarize fitted model
    print(fitted_model.summary())
    
    
    # Time series analysis
    print(fitted_model.y_hat) # in sample prediction
    print(fitted_model.resid) # in sample residuals
    print(fitted_model.aic)
    # Reading model parameters
    print(fitted_model.params.alpha)
    print(fitted_model.params.beta)
    print(fitted_model.params.x0)
    print(fitted_model.params.components.use_box_cox)
    print(fitted_model.params.components.seasonal_harmonics)