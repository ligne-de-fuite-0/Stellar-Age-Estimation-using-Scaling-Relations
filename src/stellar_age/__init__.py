"""
Stellar Age Estimation using Scaling Relations

A Python package for estimating stellar ages using asteroseismic scaling relations.
"""

__version__ = "1.0.0"
__author__ = "ligne-de-fuite-0"

from .scaling_relations import StellarAgeEstimator
from .data import load_sample_data, SOLAR_PARAMETERS
from .fitting import MCMCFitter, LeastSquaresFitter
from .validation import cross_validate, calculate_metrics
from .visualization import plot_posterior, plot_residuals, plot_corner

__all__ = [
    "StellarAgeEstimator",
    "load_sample_data",
    "SOLAR_PARAMETERS",
    "MCMCFitter",
    "LeastSquaresFitter",
    "cross_validate",
    "calculate_metrics",
    "plot_posterior",
    "plot_residuals",
    "plot_corner",
]