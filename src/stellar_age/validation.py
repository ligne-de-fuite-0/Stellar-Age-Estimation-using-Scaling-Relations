"""
Model validation and performance metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from .scaling_relations import StellarAgeEstimator

def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    dict
        Dictionary containing various metrics
    """
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Relative errors
    relative_errors = np.abs((y_pred - y_true) / y_true)
    mean_relative_error = np.mean(relative_errors)
    median_relative_error = np.median(relative_errors)
    
    # Residuals
    residuals = y_pred - y_true
    residual_std = np.std(residuals)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_relative_error': mean_relative_error,
        'median_relative_error': median_relative_error,
        'residual_std': residual_std,
        'relative_errors': relative_errors,
        'residuals': residuals
    }

def cross_validate(data, method='mcmc', n_folds=None, **fit_kwargs):
    """
    Perform cross-validation on the stellar age estimation model.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Dataset with stellar parameters and ages
    method : str
        Fitting method ('mcmc' or 'least_squares')
    n_folds : int, optional
        Number of folds. If None, performs leave-one-out CV
    **fit_kwargs
        Additional arguments for fitting method
        
    Returns
    -------
    dict
        Cross-validation results
    """
    n_stars = len(data)
    
    if n_folds is None:
        # Leave-one-out cross-validation
        n_folds = n_stars
        
    # Initialize arrays for predictions and true values
    y_true = []
    y_pred = []
    
    if n_folds == n_stars:
        # Leave-one-out CV
        print("Performing leave-one-out cross-validation...")
        
        for i in range(n_stars):
            # Split data
            train_data = data.drop(data.index[i])
            test_data = data.iloc[i:i+1]
            
            # Fit model
            estimator = StellarAgeEstimator()
            
            if method == 'mcmc':
                estimator.fit_mcmc(train_data, **fit_kwargs)
            elif method == 'least_squares':
                estimator.fit_least_squares(train_data, **fit_kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Predict
            pred_age = estimator.predict_ages(test_data)[0]
            true_age = test_data['age'].iloc[0]
            
            y_pred.append(pred_age)
            y_true.append(true_age)
            
            if (i + 1) % 50 == 0:
                print(f"Completed {i + 1}/{n_stars} folds")
    
    else:
        # K-fold cross-validation
        print(f"Performing {n_folds}-fold cross-validation...")
        
        fold_size = n_stars // n_folds
        indices = np.random.permutation(n_stars)
        
        for fold in range(n_folds):
            # Define test indices
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_stars
            test_indices = indices[start_idx:end_idx]
            train_indices = np.setdiff1d(indices, test_indices)
            
            # Split data
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
            
            # Fit model
            estimator = StellarAgeEstimator()
            
            if method == 'mcmc':
                estimator.fit_mcmc(train_data, **fit_kwargs)
            elif method == 'least_squares':
                estimator.fit_least_squares(train_data, **fit_kwargs)
            
            # Predict
            pred_ages = estimator.predict_ages(test_data)
            true_ages = test_data['age'].values
            
            y_pred.extend(pred_ages)
            y_true.extend(true_ages)
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metrics = calculate_metrics(y_true, y_pred)
    
    # Add cross-validation specific results
    cv_results = {
        'method': method,
        'n_folds': n_folds,
        'y_true': y_true,
        'y_pred': y_pred,
        **metrics
    }
    
    print(f"Cross-validation completed!")
    print(f"Mean relative error: {metrics['mean_relative_error']:.1%}")
    print(f"R² score: {metrics['r2']:.3f}")
    
    return cv_results

def bootstrap_uncertainty(data, estimator, n_bootstrap=1000):
    """
    Estimate parameter uncertainties using bootstrap resampling.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Training data
    estimator : StellarAgeEstimator
        Fitted estimator
    n_bootstrap : int
        Number of bootstrap samples
        
    Returns
    -------
    dict
        Bootstrap results with parameter uncertainties
    """
    n_stars = len(data)
    bootstrap_params = []
    
    print(f"Running {n_bootstrap} bootstrap samples...")
    
    for i in range(n_bootstrap):
        # Resample data with replacement
        bootstrap_indices = np.random.choice(n_stars, size=n_stars, replace=True)
        bootstrap_data = data.iloc[bootstrap_indices]
        
        # Fit estimator
        boot_estimator = StellarAgeEstimator()
        boot_estimator.fit_least_squares(bootstrap_data)
        
        bootstrap_params.append([
            boot_estimator.exponents['alpha'],
            boot_estimator.exponents['beta'],
            boot_estimator.exponents['gamma'],
            boot_estimator.exponents['delta'],
            boot_estimator.exponents['eta']
        ])
        
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{n_bootstrap} bootstrap samples")
    
    bootstrap_params = np.array(bootstrap_params)
    
    # Calculate statistics
    param_names = ['alpha', 'beta', 'gamma', 'delta', 'eta']
    uncertainties = {}
    
    for i, param in enumerate(param_names):
        uncertainties[param] = {
            'mean': np.mean(bootstrap_params[:, i]),
            'std': np.std(bootstrap_params[:, i]),
            'percentile_16': np.percentile(bootstrap_params[:, i], 16),
            'percentile_84': np.percentile(bootstrap_params[:, i], 84),
        }
    
    return {
        'bootstrap_params': bootstrap_params,
        'uncertainties': uncertainties,
        'param_names': param_names
    }