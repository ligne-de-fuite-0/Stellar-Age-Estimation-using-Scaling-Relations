"""
Visualization tools for stellar age estimation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner
from matplotlib.patches import Rectangle

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_posterior(posterior_samples, param_names=None, title="Posterior Distributions"):
    """
    Plot posterior distributions from MCMC fitting.
    
    Parameters
    ----------
    posterior_samples : numpy.ndarray
        MCMC posterior samples
    param_names : list, optional
        Parameter names for labeling
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    if param_names is None:
        param_names = ['α', 'β', 'γ', 'δ', 'η']
    
    # Create corner plot
    fig = corner.corner(
        posterior_samples,
        labels=param_names,
        truths=None,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        labelpad=0.3,
    )
    
    fig.suptitle(title, fontsize=16, y=0.98)
    
    return fig

def plot_residuals(y_true, y_pred, title="Residual Analysis"):
    """
    Plot residual analysis for age predictions.
    
    Parameters
    ----------
    y_true : array-like
        True stellar ages
    y_pred : array-like
        Predicted stellar ages
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Predicted vs True
    ax1.scatter(y_true, y_pred, alpha=0.6, s=30)
    min_age = min(min(y_true), min(y_pred))
    max_age = max(max(y_true), max(y_pred))
    ax1.plot([min_age, max_age], [min_age, max_age], 'r--', lw=2, label='Perfect fit')
    ax1.set_xlabel('True Age (Gyr)')
    ax1.set_ylabel('Predicted Age (Gyr)')
    ax1.set_title('Predicted vs True Ages')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_pred / y_true
    ax2.scatter(y_true, residuals, alpha=0.6, s=30)
    ax2.axhline(y=1, color='r', linestyle='--', lw=2, label='Perfect fit')
    ax2.set_xlabel('True Age (Gyr)')
    ax2.set_ylabel('Predicted / True Age')
    ax2.set_title('Relative Residuals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mse = np.mean((y_pred - y_true)**2)
    r2 = 1 - np.sum((y_pred - y_true)**2) / np.sum((y_true - np.mean(y_true))**2)
    mean_rel_error = np.mean(np.abs((y_pred - y_true) / y_true))
    
    stats_text = f'R² = {r2:.3f}\nMSE = {mse:.3f}\nMean Rel. Error = {mean_rel_error:.1%}'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=0.98)
    
    return fig

def plot_parameter_effects(data, estimator, param_name='nu_max'):
    """
    Plot the effect of a specific parameter on age predictions.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Stellar data
    estimator : StellarAgeEstimator
        Fitted estimator
    param_name : str
        Parameter to analyze
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate residuals
    y_pred = estimator.predict_ages(data)
    y_true = data['age'].values
    residuals = y_pred - y_true
    
    # Plot
    param_values = data[param_name].values
    ax.scatter(param_values, residuals, alpha=0.6, s=30)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel(f'{param_name}')
    ax.set_ylabel('Residual (Predicted - True) [Gyr]')
    ax.set_title(f'Age Prediction Residuals vs {param_name}')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_corner(posterior_samples, param_names=None):
    """
    Create a corner plot of posterior distributions.
    
    Parameters
    ----------
    posterior_samples : numpy.ndarray
        MCMC posterior samples
    param_names : list, optional
        Parameter names
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    if param_names is None:
        param_names = ['α', 'β', 'γ', 'δ', 'η']
    
    fig = corner.corner(
        posterior_samples,
        labels=param_names,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        hist_kwargs={'density': True},
        plot_datapoints=False,
        fill_contours=True,
        levels=[0.68, 0.95],
    )
    
    return fig

def plot_age_distribution(data, title="Stellar Age Distribution"):
    """
    Plot distribution of stellar ages.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Stellar data
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(data['age'], bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Stellar Age (Gyr)')
    ax.set_ylabel('Number of Stars')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_age = np.mean(data['age'])
    median_age = np.median(data['age'])
    ax.axvline(mean_age, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_age:.2f} Gyr')
    ax.axvline(median_age, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_age:.2f} Gyr')
    ax.legend()
    
    return fig

def plot_hr_diagram(data, title="Hertzsprung-Russell Diagram"):
    """
    Plot H-R diagram colored by stellar age.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Stellar data
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Calculate approximate luminosity from nu_max and teff
    # This is a simplified approach
    log_g = np.log10(data['nu_max'] / 3090) + 0.5 * np.log10(data['teff'] / 5777) + 4.44
    
    # Color by age
    scatter = ax.scatter(data['teff'], log_g, c=data['age'], 
                        cmap='viridis_r', s=50, alpha=0.7)
    
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('Effective Temperature (K)')
    ax.set_ylabel('log g (cm/s²)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Stellar Age (Gyr)')
    
    return fig