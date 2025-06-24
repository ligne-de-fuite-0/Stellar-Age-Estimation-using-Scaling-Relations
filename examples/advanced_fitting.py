#!/usr/bin/env python3
"""
Advanced fitting example using MCMC and cross-validation.
"""

import numpy as np
import pandas as pd
from stellar_age import StellarAgeEstimator, load_sample_data
from stellar_age.validation import cross_validate, bootstrap_uncertainty
from stellar_age.visualization import plot_posterior, plot_corner

def main():
    print("=== Advanced Stellar Age Estimation ===\n")
    
    # 1. Load data
    print("1. Loading sample data...")
    data = load_sample_data()
    print(f"   Loaded {len(data)} stars")
    
    # 2. Fit using MCMC
    print("\n2. Fitting model using MCMC...")
    estimator = StellarAgeEstimator()
    
    # Use fewer steps for demonstration
    estimator.fit_mcmc(
        data, 
        n_walkers=16, 
        n_steps=1000, 
        burn_in=200,
        progress=True
    )
    
    print(f"   Fitted exponents: {estimator.exponents}")
    
    # 3. Analyze posterior
    print("\n3. Analyzing posterior distributions...")
    posterior = estimator.get_posterior()
    
    param_names = ['α', 'β', 'γ', 'δ', 'η']
    for i, param in enumerate(param_names):
        values = posterior[:, i]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"   {param}: {mean_val:.3f} ± {std_val:.3f}")
    
    # 4. Create posterior plots
    print("\n4. Creating posterior visualizations...")
    
    # Corner plot
    fig_corner = plot_corner(posterior, param_names)
    fig_corner.savefig('advanced_corner_plot.png', dpi=300, bbox_inches='tight')
    
    # Posterior distribution plot
    fig_posterior = plot_posterior(posterior, param_names)
    fig_posterior.savefig('advanced_posterior.png', dpi=300, bbox_inches='tight')
    
    print("   Saved plots: 'advanced_corner_plot.png', 'advanced_posterior.png'")
    
    # 5. Uncertainty estimation for a single star
    print("\n5. Calculating uncertainties for sample star...")
    sample_star = {
        'nu_max': 793.0,
        'delta_nu': 45.3, 
        'delta_nu_small': 4.44,
        'teff': 5257.0,
        'feh': -0.14
    }
    
    uncertainties = estimator.calculate_uncertainties(sample_star, n_samples=500)
    print(f"   Age: {uncertainties['mean']:.2f} ± {uncertainties['std']:.2f} Gyr")
    print(f"   68% confidence interval: [{uncertainties['percentile_16']:.2f}, {uncertainties['percentile_84']:.2f}] Gyr")
    
    # 6. Cross-validation (subset for speed)
    print("\n6. Performing cross-validation (50 stars)...")
    subset_data = data.sample(n=50, random_state=42)
    
    cv_results = cross_validate(
        subset_data, 
        method='least_squares',  # Faster than MCMC for CV
        n_folds=5  # 5-fold instead of leave-one-out for speed
    )
    
    print(f"   CV Mean relative error: {cv_results['mean_relative_error']:.1%}")
    print(f"   CV R² score: {cv_results['r2']:.3f}")
    
    # 7. Bootstrap uncertainty estimation (subset)
    print("\n7. Bootstrap uncertainty estimation (subset)...")
    bootstrap_results = bootstrap_uncertainty(
        subset_data, 
        estimator, 
        n_bootstrap=100  # Reduced for demonstration
    )
    
    print("   Bootstrap parameter uncertainties:")
    for param, uncertainty in bootstrap_results['uncertainties'].items():
        print(f"   {param}: {uncertainty['mean']:.3f} ± {uncertainty['std']:.3f}")

if __name__ == "__main__":
    main()