# Stellar Age Estimation using Scaling Relations

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)

A Python package for estimating stellar ages using asteroseismic scaling relations based on the research paper "Stellar age estimation based on stellar parameters' scaling relation".

## Overview

This package implements a novel scaling relation for estimating the ages of solar-like stars using five key asteroseismic and stellar parameters:

- Large frequency separation (Δν)
- Maximum oscillation power frequency (νmax)
- Small frequency separation (δν)
- Effective temperature (Teff)
- Metallicity ([Fe/H])

The method achieves ~10.8% accuracy in age estimation for solar-like main-sequence stars.

## Features

- **Scaling Relations**: Implementation of the improved stellar age scaling relation
- **Fitting Methods**: Both least-squares and MCMC (Bayesian) parameter fitting
- **Cross-Validation**: Leave-one-out cross-validation for model assessment
- **Visualization**: Comprehensive plotting tools for analysis and results
- **Sample Data**: 209 solar-like stars with validated parameters

## Installation

### From PyPI (recommended)
```bash
pip install stellar-age-estimation
```

### From Source
```bash
git clone https://github.com/ligne-de-fuite-0/stellar-age-estimation.git
cd stellar-age-estimation
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/ligne-de-fuite-0/stellar-age-estimation.git
cd stellar-age-estimation
pip install -e .[dev,notebooks]
```

## Quick Start

```python
import numpy as np
from stellar_age import StellarAgeEstimator, load_sample_data

# Load sample data
data = load_sample_data()

# Initialize the estimator
estimator = StellarAgeEstimator()

# Fit the scaling relation using MCMC
estimator.fit_mcmc(data)

# Estimate age for a single star
star_params = {
    'nu_max': 793.0,  # μHz
    'delta_nu': 45.3,  # μHz
    'delta_nu_small': 4.44,  # μHz
    'teff': 5257.0,  # K
    'feh': -0.14  # dex
}

estimated_age = estimator.predict_age(star_params)
print(f"Estimated stellar age: {estimated_age:.2f} Gyr")
```

## Documentation

- [Full Documentation](https://stellar-age-estimation.readthedocs.io/)
- [API Reference](docs/api_reference.md)
- [Methodology](docs/methodology.md)
- [Examples and Tutorials](notebooks/)

## Examples

### Basic Usage
```python
from stellar_age import StellarAgeEstimator

# Create estimator with pre-fitted parameters
estimator = StellarAgeEstimator.from_paper_results()

# Estimate age for multiple stars
ages = estimator.predict_ages(star_data)
```

### Advanced Fitting
```python
# Custom MCMC fitting
estimator = StellarAgeEstimator()
estimator.fit_mcmc(
    data, 
    n_walkers=32, 
    n_steps=5000, 
    burn_in=500
)

# Access posterior distributions
posterior = estimator.get_posterior()
```

### Cross-Validation
```python
from stellar_age.validation import cross_validate

# Perform leave-one-out cross-validation
cv_results = cross_validate(data, method='mcmc')
print(f"Mean relative error: {cv_results['mean_error']:.1%}")
```

## Scientific Background

This implementation is based on the scaling relation:

```
Y/Y☉ = (νmax/νmax,☉)^α × (Δν/Δν☉)^β × (δν/δν☉)^γ × (Teff/Teff,☉)^δ × exp([Fe/H])^η
```

Where Y is the stellar age and the fitted exponents are:
- α = -7.017
- β = 9.578  
- γ = -4.202
- δ = -0.140
- η = -1.254

## Data

The package includes a curated dataset of 209 solar-like stars with precisely determined parameters from the Kepler mission and ground-based spectroscopy.

## Citation

If you use this package in your research, please cite:

```bibtex
@article{stellar_age_2024,
    title={Stellar age estimation based on stellar parameters' scaling relation},
    author={Your Name},
    journal={Journal Name},
    year={2024},
    volume={XX},
    pages={XXX-XXX}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kepler mission for providing high-quality asteroseismic data
- KASC (Kepler Asteroseismic Science Consortium) for stellar parameters
- The emcee team for the MCMC implementation
- The broader asteroseismology community

## Support

- 📧 Email: your.email@example.com
- 🐛 [Issue Tracker](https://github.com/ligne-de-fuite-0/stellar-age-estimation/issues)
- 💬 [Discussions](https://github.com/ligne-de-fuite-0/stellar-age-estimation/discussions)
