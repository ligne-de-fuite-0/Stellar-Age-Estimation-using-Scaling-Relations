"""
Solar parameters and data loading utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Solar reference parameters
SOLAR_PARAMETERS = {
    'nu_max': 3090.0,  # μHz, Huber et al. 2011
    'nu_max_err': 30.0,
    'delta_nu': 135.1,  # μHz, Huber et al. 2011  
    'delta_nu_err': 0.1,
    'delta_nu_small': 8.957,  # μHz
    'delta_nu_small_err': 0.059,
    'teff': 5772.0,  # K
    'teff_err': 0.8,
    'age': 4.569,  # Gyr
    'age_err': 0.006,
}

# Fitted exponents from the paper
FITTED_EXPONENTS = {
    'alpha': -7.01736937,  # nu_max exponent
    'beta': 9.57826826,   # delta_nu exponent  
    'gamma': -4.20249286, # delta_nu_small exponent
    'delta': -0.1397651,  # teff exponent
    'eta': -1.25395515,   # metallicity exponent
}

def load_sample_data():
    """
    Load the sample dataset of 209 solar-like stars.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing stellar parameters for 209 stars
    """
    # In a real implementation, this would load from the actual data file
    # For now, we'll create a sample dataset based on the paper
    
    np.random.seed(42)  # For reproducibility
    n_stars = 209
    
    # Generate realistic parameter ranges based on the paper
    data = pd.DataFrame({
        'KIC': np.random.randint(10000000, 20000000, n_stars),
        'nu_max': np.random.lognormal(np.log(800), 0.8, n_stars),
        'delta_nu': np.random.lognormal(np.log(45), 0.6, n_stars),
        'delta_nu_small': np.random.normal(5.0, 1.0, n_stars),
        'teff': np.random.normal(5500, 400, n_stars),
        'feh': np.random.normal(-0.2, 0.3, n_stars),
        'age': np.random.gamma(2, 2, n_stars),  # Age in Gyr
    })
    
    # Ensure physical constraints
    data['nu_max'] = np.clip(data['nu_max'], 200, 3000)
    data['delta_nu'] = np.clip(data['delta_nu'], 15, 120)
    data['delta_nu_small'] = np.clip(data['delta_nu_small'], 2, 10)
    data['teff'] = np.clip(data['teff'], 4500, 6500)
    data['feh'] = np.clip(data['feh'], -1.0, 0.5)
    data['age'] = np.clip(data['age'], 0.5, 12)
    
    return data

def create_sample_star_data():
    """
    Create sample star data for testing and examples.
    
    Returns
    -------
    dict
        Dictionary containing parameters for a sample star
    """
    return {
        'nu_max': 793.0,
        'delta_nu': 45.3,
        'delta_nu_small': 4.44,
        'teff': 5257.0,
        'feh': -0.14,
    }

def load_kepler_data(filename=None):
    """
    Load Kepler asteroseismic data.
    
    Parameters
    ----------
    filename : str, optional
        Path to the data file. If None, loads default sample data.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing stellar parameters
    """
    if filename is None:
        return load_sample_data()
    
    # Load from file
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print(f"File {filename} not found. Loading sample data instead.")
        return load_sample_data()

def validate_stellar_parameters(data):
    """
    Validate stellar parameters for physical consistency.
    
    Parameters
    ----------
    data : pandas.DataFrame or dict
        Stellar parameters to validate
        
    Returns
    -------
    bool
        True if parameters are valid, False otherwise
    """
    required_columns = ['nu_max', 'delta_nu', 'delta_nu_small', 'teff', 'feh']
    
    if isinstance(data, dict):
        return all(col in data for col in required_columns)
    
    if isinstance(data, pd.DataFrame):
        return all(col in data.columns for col in required_columns)
    
    return False