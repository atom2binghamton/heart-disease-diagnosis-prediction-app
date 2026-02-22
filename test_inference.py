"""
Test script to verify saved Heart Disease classification models.
Strictly follows the Top-3 feature schema + patient_id logic.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def test_model(model_path):
    """Load model and test classification inference."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print('='*60)
    
    try:
        # Load model
        print("Loading model...", end=" ")
        model = joblib.load(model_path)
        print("✓")
        
        if hasattr(model, 'named_steps'):
            print(f"Pipeline steps: {list(model.named_steps.keys())}")
        
        # Create sample data following the simplified schema
        # We include patient_id to satisfy the ColumnDropper in the pipeline
        sample_data = pd.DataFrame({
            'patient_id': [9001, 9002, 9003],
            'ca': [0.0, 1.0, 2.0],        # Number of major vessels
            'cp_id': [3.0, 2.0, 1.0],     # Chest pain type IDs
            'thal_id': [1.0, 2.0, 3.0],   # Thalassemia IDs
        })
        
        # Predict
        print("Running inference...", end=" ")
        # .predict gives the class (0 or 1)
        predictions = model.predict(sample_data)
        # .predict_proba gives the confidence scores
        probabilities = model.predict_proba(sample_data)[:, 1] 
        print("✓")
        
        # Show results
        print("\nClassification Results:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities), 1):
            status = "POSITIVE" if pred == 1 else "NEGATIVE"
            print(f"  Sample {i}: {status} (Confidence: {prob:.2%})")
        
        print(f"\n✓ {Path(model_path).name} - SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        # Hint for common error: check if heart_pipeline.py is in the same directory
        if "ColumnDropper" in str(e):
            print("  HINT: ColumnDropper class not found. Ensure heart_pipeline.py is accessible.")
        return False

def main():
    """Test both heart disease models."""
    # Ensure these paths match your saved filenames from previous steps
    models = [
        "models/global_best_heart_model.pkl",
        "models/global_best_heart_optuna.pkl"
    ]
    
    print("Testing Heart Disease Classification Models...")
    results = []
    for m in models:
        if Path(m).exists():
            results.append(test_model(m))
        else:
            print(f"\n! Skipping: {m} (File not found)")
    
    if not results:
        print("\n✗ No model files found in the models/ directory.")
        return False

    print(f"\n{'='*60}")
    print(f"SUMMARY: {sum(results)}/{len(results)} models passed")
    print('='*60)
    
    return all(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)