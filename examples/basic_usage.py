#!/usr/bin/env python3
"""
Basic usage example for stellar age estimation.
"""

import numpy as np
import pandas as pd
from stellar_age import StellarAgeEstimator, load_sample_data, create_sample_star_data
from stellar_age.visualization import plot_residuals

def main():
    print("=== Stellar Age Estimation - Basic Usage ===\n")
    
    # 1. Load sample data
    print("1. Loading sample data...")
    data = load_sample_data()
    print(f"   Loaded {len(data)} stars")
    print(f"   Sample parameters: {list(data.columns)}")
    
    # 2. Create estimator with pre-fitted parameters
    print("\n2. Creating estimator with paper results...")
    estimator = StellarAgeEstimator.from_paper_results()
    print(f"   Fitted exponents: {estimator.exponents}")
    
    # 3. Predict age for a single star
    print("\n3. Predicting age for a sample star...")
    sample_star = create_sample_star_data()
    print(f"   Star parameters: {sample_star}")
    
    estimated_age = estimator.predict_age(sample_star)
    print(f"   Estimated age: {estimated_age:.2f} Gyr")
    
    # 4. Predict ages for multiple stars
    print("\n4. Predicting ages for all stars...")
    predicted_ages = estimator.predict_ages(data)
    
    # Calculate some statistics
    mean_error = np.mean(np.abs(predicted_ages - data['age']) / data['age'])
    r2 = 1 - np.sum((predicted_ages - data['age'])**2) / np.sum((data['age'] - np.mean(data['age']))**2)
    
    print(f"   Mean relative error: {mean_error:.1%}")
    print(f"   R² score: {r2:.3f}")
    
    # 5. Create visualization
    print("\n5. Creating residual plot...")
    fig = plot_residuals(data['age'], predicted_ages)
    fig.savefig('basic_usage_residuals.png', dpi=300, bbox_inches='tight')
    print("   Saved plot as 'basic_usage_residuals.png'")
    
    # 6. Show some individual predictions
    print("\n6. Sample predictions:")
    print("   KIC         True Age    Pred Age    Error")
    print("   " + "-"*42)
    
    for i in range(5):
        kic = data.iloc[i]['KIC']
        true_age = data.iloc[i]['age']
        pred_age = predicted_ages[i]
        error = abs(pred_age - true_age) / true_age * 100
        
        print(f"   {kic:>10}  {true_age:>8.2f}    {pred_age:>8.2f}    {error:>5.1f}%")

if __name__ == "__main__":
    main()