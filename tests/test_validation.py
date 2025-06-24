"""
Tests for validation methods.
"""

import pytest
import numpy as np
from stellar_age import StellarAgeEstimator, load_sample_data
from stellar_age.validation import calculate_metrics, cross_validate

class TestMetrics:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.y_true = np.array([1, 2, 3, 4, 5])
        self.y_