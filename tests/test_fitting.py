"""
Tests for fitting methods.
"""

import pytest
import numpy as np
from stellar_age import StellarAgeEstimator, load_sample_data
from stellar_age.fitting import LeastSquaresFitter, MCMCFitter

class TestLeastSquaresFitter:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = StellarAgeEstimator()
        self.fitter = LeastSquaresFitter(self.estimator)
        self.data = load_sample_data().head(50)  # Small subset for speed
    
    def test_loss_function(self):
        """Test loss function calculation."""
        params = [-7.0, 9.5, -4.2, -0.14, -1.25]
        loss = self.fitter.loss_function(params, self.data)
        
        assert isinstance(loss, (float, np.floating))
        assert loss >= 0
    
    def test_fitting(self):
        """Test least squares fitting."""
        fitted_params, result = self.fitter.fit(self.data)
        
        assert result.success
        assert all(param in fitted_params for param in 
                  ['alpha', 'beta', 'gamma', 'delta', 'eta'])

class TestMCMCFitter:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = StellarAgeEstimator()
        self.fitter = MCMCFitter(self.estimator)
        self.data = load_sample_data().head(20)  # Very small subset for speed
    
    def test_log_likelihood(self):
        """Test log likelihood calculation."""
        params = [-7.0, 9.5, -4.2, -0.14, -1.25]
        log_like = self.fitter.log_likelihood(params, self.data)
        
        assert isinstance(log_like, (float, np.floating))
        assert np.isfinite(log_like)
    
    def test_log_prior(self):
        """Test log prior calculation."""
        valid_params = [-7.0, 9.5, -4.2, -0.14, -1.25]
        invalid_params = [-50, 100, -50, 50, 50]
        
        assert self.fitter.log_prior(valid_params) == 0.0
        assert self.fitter.log_prior(invalid_params) == -np.inf
    
    def test_mcmc_fitting(self):
        """Test MCMC fitting (minimal)."""
        fitted_params, samples, sampler = self.fitter.fit(
            self.data, 
            n_walkers=8, 
            n_steps=100, 
            burn_in=20,
            progress=False
        )
        
        assert all(param in fitted_params for param in 
                  ['alpha', 'beta', 'gamma', 'delta', 'eta'])
        assert samples.shape[1] == 5  # 5 parameters
        assert len(samples) > 0