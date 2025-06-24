"""
Core stellar age scaling relations implementation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .data import SOLAR_PARAMETERS, FITTED_EXPONENTS, validate_stellar_parameters
from .fitting import MCMCFitter, LeastSquaresFitter

class StellarAgeEstimator:
    """
    Stellar age estimator using asteroseismic scaling relations.
    
    This class implements the scaling relation for estimating stellar ages
    based on five key parameters: nu_max, delta_nu, delta_nu_small, teff, and [Fe/H].
    
    Parameters
    ----------
    solar_params : dict, optional
        Solar reference parameters. If None, uses default values.
    exponents : dict, optional
        Scaling relation exponents. If None, uses fitted values from paper.
    """
    
    def __init__(self, solar_params=None, exponents=None):
        self.solar_params = solar_params or SOLAR_PARAMETERS
        self.exponents = exponents or FITTED_EXPONENTS
        self.is_fitted = exponents is not None
        self.posterior_samples = None
        
    @classmethod
    def from_paper_results(cls):
        """
        Create estimator with pre-fitted parameters from the paper.
        
        Returns
        -------
        StellarAgeEstimator
            Estimator with fitted parameters
        """
        return cls(exponents=FITTED_EXPONENTS)
    
    def scaling_relation(self, nu_max, delta_nu, delta_nu_small, teff, feh, 
                        alpha=None, beta=None, gamma=None, delta=None, eta=None):
        """
        Calculate stellar age using the scaling relation.
        
        Y/Y☉ = (νmax/νmax,☉)^α × (Δν/Δν☉)^β × (δν/δν☉)^γ × (Teff/Teff,☉)^δ × exp([Fe/H])^η
        
        Parameters
        ----------
        nu_max : float or array
            Maximum oscillation frequency in μHz
        delta_nu : float or array
            Large frequency separation in μHz
        delta_nu_small : float or array
            Small frequency separation in μHz
        teff : float or array
            Effective temperature in K
        feh : float or array
            Metallicity [Fe/H] in dex
        alpha, beta, gamma, delta, eta : float, optional
            Scaling relation exponents. If None, uses fitted values.
            
        Returns
        -------
        float or array
            Estimated stellar age in Gyr
        """
        # Use fitted exponents if not provided
        alpha = alpha or self.exponents['alpha']
        beta = beta or self.exponents['beta']
        gamma = gamma or self.exponents['gamma']
        delta = delta or self.exponents['delta']
        eta = eta or self.exponents['eta']
        
        # Solar parameters
        nu_max_sun = self.solar_params['nu_max']
        delta_nu_sun = self.solar_params['delta_nu']
        delta_nu_small_sun = self.solar_params['delta_nu_small']
        teff_sun = self.solar_params['teff']
        age_sun = self.solar_params['age']
        
        # Calculate scaled ratios
        nu_max_ratio = nu_max / nu_max_sun
        delta_nu_ratio = delta_nu / delta_nu_sun
        delta_nu_small_ratio = delta_nu_small / delta_nu_small_sun
        teff_ratio = teff / teff_sun
        
        # Apply scaling relation
        age_ratio = (nu_max_ratio**alpha * 
                    delta_nu_ratio**beta * 
                    delta_nu_small_ratio**gamma * 
                    teff_ratio**delta * 
                    np.exp(feh)**eta)
        
        return age_ratio * age_sun
    
    def predict_age(self, star_params):
        """
        Predict age for a single star.
        
        Parameters
        ----------
        star_params : dict
            Dictionary containing stellar parameters
            
        Returns
        -------
        float
            Estimated stellar age in Gyr
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if not validate_stellar_parameters(star_params):
            raise ValueError("Invalid stellar parameters")
            
        return self.scaling_relation(
            star_params['nu_max'],
            star_params['delta_nu'], 
            star_params['delta_nu_small'],
            star_params['teff'],
            star_params['feh']
        )
    
    def predict_ages(self, data):
        """
        Predict ages for multiple stars.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing stellar parameters
            
        Returns
        -------
        numpy.ndarray
            Estimated stellar ages in Gyr
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if not validate_stellar_parameters(data):
            raise ValueError("Invalid stellar parameters")
            
        return self.scaling_relation(
            data['nu_max'].values,
            data['delta_nu'].values,
            data['delta_nu_small'].values, 
            data['teff'].values,
            data['feh'].values
        )
    
    def fit_least_squares(self, data):
        """
        Fit scaling relation exponents using least squares method.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Training data with stellar parameters and ages
        """
        fitter = LeastSquaresFitter(self)
        self.exponents, self.fit_results = fitter.fit(data)
        self.is_fitted = True
        
    def fit_mcmc(self, data, n_walkers=32, n_steps=5000, burn_in=500, 
                 progress=True):
        """
        Fit scaling relation exponents using MCMC method.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Training data with stellar parameters and ages
        n_walkers : int, optional
            Number of MCMC walkers
        n_steps : int, optional
            Number of MCMC steps
        burn_in : int, optional
            Number of burn-in steps to discard
        progress : bool, optional
            Show progress bar
        """
        fitter = MCMCFitter(self)
        self.exponents, self.posterior_samples, self.fit_results = fitter.fit(
            data, n_walkers=n_walkers, n_steps=n_steps, 
            burn_in=burn_in, progress=progress
        )
        self.is_fitted = True
        
    def get_posterior(self):
        """
        Get posterior samples from MCMC fitting.
        
        Returns
        -------
        numpy.ndarray or None
            Posterior samples if MCMC was used, None otherwise
        """
        return self.posterior_samples
    
    def calculate_uncertainties(self, star_params, n_samples=1000):
        """
        Calculate age uncertainties using posterior samples.
        
        Parameters
        ----------
        star_params : dict
            Dictionary containing stellar parameters
        n_samples : int, optional
            Number of posterior samples to use
            
        Returns
        -------
        dict
            Dictionary with mean age and uncertainties
        """
        if self.posterior_samples is None:
            raise ValueError("Posterior samples not available. Run MCMC fitting first.")
            
        # Sample from posterior
        n_total = len(self.posterior_samples)
        indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
        samples = self.posterior_samples[indices]
        
        ages = []
        for sample in samples:
            alpha, beta, gamma, delta, eta = sample
            age = self.scaling_relation(
                star_params['nu_max'],
                star_params['delta_nu'],
                star_params['delta_nu_small'], 
                star_params['teff'],
                star_params['feh'],
                alpha=alpha, beta=beta, gamma=gamma, 
                delta=delta, eta=eta
            )
            ages.append(age)
        
        ages = np.array(ages)
        
        return {
            'mean': np.mean(ages),
            'std': np.std(ages),
            'median': np.median(ages),
            'percentile_16': np.percentile(ages, 16),
            'percentile_84': np.percentile(ages, 84),
        }