"""
Tests for stellar age scaling relations.
"""

import pytest
import numpy as np
import pandas as pd
from stellar_age import StellarAgeEstimator, load_sample_data, SOLAR_PARAMETERS

class TestStellarAgeEstimator:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = StellarAgeEstimator.from_paper_results()
        self.sample_data = load_sample_data()
        self.sample_star = {
            'nu_max': 793.0,
            'delta_nu': 45.3,
            'delta_nu_small': 4.44,
            'teff': 5257.0,
            'feh': -0.14
        }
    
    def test_solar_scaling(self):
        """Test that solar parameters give solar age."""
        solar_age = self.estimator.scaling_relation(
            SOLAR_PARAMETERS['nu_max'],
            SOLAR_PARAMETERS['delta_nu'],
            SOLAR_PARAMETERS['delta_nu_small'],
            SOLAR_PARAMETERS['teff'],
            0.0  # Solar metallicity
        )
        
        # Should be close to solar age
        assert abs(solar_age - SOLAR_PARAMETERS['age']) < 0.5
    
    def test_predict_age_single(self):
        """Test single star age prediction."""
        age = self.estimator.predict_age(self.sample_star)
        
        assert isinstance(age, (float, np.floating))
        assert 0.1 < age < 15.0  # Reasonable age range
    
    def test_predict_ages_multiple(self):
        """Test multiple star age predictions."""
        ages = self.estimator.predict_ages(self.sample_data)
        
        assert len(ages) == len(self.sample_data)
        assert all(0.1 < age < 15.0 for age in ages)
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        invalid_star = {'invalid': 'params'}
        
        with pytest.raises(ValueError):
            self.estimator.predict_age(invalid_star)
    
    def test_fitting_least_squares(self):
        """Test least squares fitting."""
        estimator = StellarAgeEstimator()
        estimator.fit_least_squares(self.sample_data.head(50))
        
        assert estimator.is_fitted
        assert all(param in estimator.exponents for param in 
                  ['alpha', 'beta', 'gamma', 'delta', 'eta'])
    
    def test_parameter_ranges(self):
        """Test that fitted parameters are in reasonable ranges."""
        assert -15 < self.estimator.exponents['alpha'] < 0
        assert 0 < self.estimator.exponents['beta'] < 20
        assert -10 < self.estimator.exponents['gamma'] < 0
        assert -5 < self.estimator.exponents['delta'] < 5
        assert -5 < self.estimator.exponents['eta'] < 5

class TestDataValidation:
    
    def test_load_sample_data(self):
        """Test sample data loading."""
        data = load_sample_data()
        
        required_columns = ['nu_max', 'delta_nu', 'delta_nu_small', 'teff', 'feh', 'age']
        assert all(col in data.columns for col in required_columns)
        assert len(data) > 0
    
    def test_parameter_validation(self):
        """Test parameter validation function."""
        from stellar_age.data import validate_stellar_parameters
        
        valid_params = {
            'nu_max': 800,
            'delta_nu': 45,
            'delta_nu_small': 4.5,
            'teff': 5500,
            'feh': -0.1
        }
        
        invalid_params = {'invalid': 'params'}
        
        assert validate_stellar_parameters(valid_params)
        assert not validate_stellar_parameters(invalid_params)