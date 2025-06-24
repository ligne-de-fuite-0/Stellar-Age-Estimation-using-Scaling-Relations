"""
Parameter fitting methods for stellar age scaling relations.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import emcee
from tqdm import tqdm

class LeastSquaresFitter:
    """
    Least squares fitting for scaling relation parameters.
    """
    
    def __init__(self, estimator):
        self.estimator = estimator
        
    def loss_function(self, params, data):
        """
        Calculate loss function for least squares fitting.
        
        Parameters
        ----------
        params : array-like
            Model parameters [alpha, beta, gamma, delta, eta]
        data : pandas.DataFrame
            Training data
            
        Returns
        -------
        float
            Sum of squared residuals
        """
        alpha, beta, gamma, delta, eta = params
        
        predicted_ages = self.estimator.scaling_relation(
            data['nu_max'].values,
            data['delta_nu'].values,
            data['delta_nu_small'].values,
            data['teff'].values,
            data['feh'].values,
            alpha=alpha, beta=beta, gamma=gamma,
            delta=delta, eta=eta
        )
        
        residuals = predicted_ages - data['age'].values
        return np.sum(residuals**2)
    
    def fit(self, data, initial_guess=None):
        """
        Fit parameters using least squares method.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Training data with stellar parameters and ages
        initial_guess : array-like, optional
            Initial parameter guess
            
        Returns
        -------
        tuple
            (fitted_parameters_dict, optimization_result)
        """
        if initial_guess is None:
            initial_guess = [-7.0, 9.5, -4.2, -0.14, -1.25]
            
        # Optimize
        result = minimize(
            self.loss_function, 
            initial_guess, 
            args=(data,),
            method='L-BFGS-B'
        )
        
        # Extract fitted parameters
        alpha, beta, gamma, delta, eta = result.x
        fitted_params = {
            'alpha': alpha,
            'beta': beta, 
            'gamma': gamma,
            'delta': delta,
            'eta': eta
        }
        
        return fitted_params, result

class MCMCFitter:
    """
    MCMC fitting for scaling relation parameters using emcee.
    """
    
    def __init__(self, estimator):
        self.estimator = estimator
        
    def log_likelihood(self, params, data):
        """
        Calculate log likelihood for MCMC fitting.
        
        Parameters
        ----------
        params : array-like
            Model parameters [alpha, beta, gamma, delta, eta]
        data : pandas.DataFrame
            Training data
            
        Returns
        -------
        float
            Log likelihood
        """
        alpha, beta, gamma, delta, eta = params
        
        predicted_ages = self.estimator.scaling_relation(
            data['nu_max'].values,
            data['delta_nu'].values, 
            data['delta_nu_small'].values,
            data['teff'].values,
            data['feh'].values,
            alpha=alpha, beta=beta, gamma=gamma,
            delta=delta, eta=eta
        )
        
        # Assume Gaussian errors
        sigma = 0.5  # Age uncertainty in Gyr
        chi2 = np.sum((predicted_ages - data['age'].values)**2 / sigma**2)
        
        return -0.5 * chi2
    
    def log_prior(self, params):
        """
        Calculate log prior probability.
        
        Parameters
        ----------
        params : array-like
            Model parameters [alpha, beta, gamma, delta, eta]
            
        Returns
        -------
        float
            Log prior probability
        """
        alpha, beta, gamma, delta, eta = params
        
        # Set reasonable bounds
        if (-15 < alpha < 0 and 
            0 < beta < 20 and 
            -10 < gamma < 0 and
            -5 < delta < 5 and
            -5 < eta < 5):
            return 0.0
        return -np.inf
    
    def log_posterior(self, params, data):
        """
        Calculate log posterior probability.
        
        Parameters
        ----------
        params : array-like
            Model parameters
        data : pandas.DataFrame
            Training data
            
        Returns
        -------
        float
            Log posterior probability
        """
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params, data)
    
    def fit(self, data, n_walkers=32, n_steps=5000, burn_in=500, 
            progress=True, initial_guess=None):
        """
        Fit parameters using MCMC method.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Training data with stellar parameters and ages
        n_walkers : int
            Number of MCMC walkers
        n_steps : int
            Number of MCMC steps
        burn_in : int
            Number of burn-in steps to discard
        progress : bool
            Show progress bar
        initial_guess : array-like, optional
            Initial parameter guess
            
        Returns
        -------
        tuple
            (fitted_parameters_dict, posterior_samples, sampler)
        """
        if initial_guess is None:
            initial_guess = [-7.0, 9.5, -4.2, -0.14, -1.25]
            
        # Set up initial positions
        ndim = len(initial_guess)
        pos = initial_guess + 1e-4 * np.random.randn(n_walkers, ndim)
        
        # Initialize sampler
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, self.log_posterior, args=(data,)
        )
        
        # Run MCMC
        if progress:
            sampler.run_mcmc(pos, n_steps, progress=True)
        else:
            sampler.run_mcmc(pos, n_steps)
        
        # Extract samples (after burn-in)
        samples = sampler.get_chain(discard=burn_in, flat=True)
        
        # Calculate best-fit parameters (median of posterior)
        alpha_fit, beta_fit, gamma_fit, delta_fit, eta_fit = np.median(samples, axis=0)
        
        fitted_params = {
            'alpha': alpha_fit,
            'beta': beta_fit,
            'gamma': gamma_fit, 
            'delta': delta_fit,
            'eta': eta_fit
        }
        
        return fitted_params, samples, sampler