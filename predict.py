"""
predict.py
--------------------------------------------------------------------------
Inference script for Celestial Object Classification & Regression.
This script:
  1. Loads the saved Random Forest Classifier, Regressor, and Target Encoder.
  2. Offers an interactive prompt for users to input custom physical parameters.
  3. Automatically runs demonstration test cases if standard input is unavailable
     or if the user requests them.
  4. Encodes and processes inputs using the same logic as the training pipeline.
  5. Outputs predicted object type, classification confidence, and estimated lifetime.
--------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Paths
MODELS_DIR = "models"
CLF_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
REG_PATH = os.path.join(MODELS_DIR, "regressor.pkl")
LE_PATH = os.path.join(MODELS_DIR, "object_encoder.pkl")

# Spectral class ordinal mapping used during training
SPECTRAL_MAPPING = {
    "NONE": 0,
    "M": 1,
    "K": 2,
    "G": 3,
    "F": 4,
    "A": 5,
    "B": 6,
    "O": 7
}

def load_ml_assets():
    """Loads saved models and label encoders from disk."""
    if not (os.path.exists(CLF_PATH) and os.path.exists(REG_PATH) and os.path.exists(LE_PATH)):
        raise FileNotFoundError(
            "Model files are missing. Please run 'python train_model.py' first to train and save the models."
        )
    
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    label_encoder = joblib.load(LE_PATH)
    
    return clf, reg, label_encoder

def process_and_predict(features_dict, clf, reg, label_encoder):
    """
    Takes raw feature inputs, processes them, runs predictions, and prints results.
    """
    # 1. Map Spectral Class
    spec_class = str(features_dict["Spectral_Class"]).strip().upper()
    spec_encoded = SPECTRAL_MAPPING.get(spec_class, 0)
    
    # 2. Build feature vector matching train order:
    # ["Mass", "Temperature", "Luminosity", "Radius", "Age", "Spectral_Class_Encoded"]
    feature_cols = ["Mass", "Temperature", "Luminosity", "Radius", "Age", "Spectral_Class_Encoded"]
    feature_df = pd.DataFrame([[
        float(features_dict["Mass"]),
        float(features_dict["Temperature"]),
        float(features_dict["Luminosity"]),
        float(features_dict["Radius"]),
        float(features_dict["Age"]),
        spec_encoded
    ]], columns=feature_cols)
    
    # 3. Classify Object Type
    class_pred_encoded = clf.predict(feature_df)[0]
    class_probs = clf.predict_proba(feature_df)[0]
    
    predicted_type = label_encoder.inverse_transform([class_pred_encoded])[0]
    confidence = class_probs[class_pred_encoded] * 100
    
    # 4. Predict Lifetime
    lifetime_pred = reg.predict(feature_df)[0]
    
    print("\n" + "="*50)
    print("                PREDICTION RESULTS                ")
    print("="*50)
    print(f"Inputs:")
    print(f"  - Mass:        {features_dict['Mass']} Solar Masses")
    print(f"  - Temperature: {features_dict['Temperature']} K")
    print(f"  - Luminosity:  {features_dict['Luminosity']} Solar Luminosity")
    print(f"  - Radius:      {features_dict['Radius']} Solar Radii")
    print(f"  - Age:         {features_dict['Age']} Billion Years")
    print(f"  - Spectral Cl: {spec_class} (Encoded as {spec_encoded})")
    print("-"*50)
    print(f"Prediction Target: Object Classification")
    print(f"  -> Predicted Type:  {predicted_type}")
    print(f"  -> Confidence:      {confidence:.2f}%")
    print("-"*50)
    print(f"Prediction Target: Stellar/Object Lifetime")
    print(f"  -> Est. Lifetime:   {lifetime_pred:.4f} Billion Years")
    print("="*50 + "\n")
    
    # Print confidence breakdowns
    print("Classification Probabilities:")
    for idx, class_name in enumerate(label_encoder.classes_):
        print(f"  - {class_name:10}: {class_probs[idx]*100:6.2f}%")
    print()

def get_float_input(prompt, default_val):
    """Safely gets float input from user with a default value."""
    try:
        user_in = input(f"{prompt} [{default_val}]: ").strip()
        if not user_in:
            return default_val
        return float(user_in)
    except (ValueError, EOFError):
        return default_val

def get_string_input(prompt, default_val, allowed_values):
    """Safely gets string input from user and validates it against allowed values."""
    try:
        user_in = input(f"{prompt} [{default_val}]: ").strip().upper()
        if not user_in:
            return default_val
        if user_in not in allowed_values:
            print(f"Invalid input! Resetting to default: {default_val}")
            return default_val
        return user_in
    except (ValueError, EOFError):
        return default_val

def run_demos(clf, reg, label_encoder):
    """Runs prediction on preset demo cases for quick verification."""
    print("\nRunning Demonstration Cases...\n")
    
    demos = [
        {
            "name": "Stellar Body (G-type Star like the Sun)",
            "data": {"Mass": 1.0, "Temperature": 5778.0, "Luminosity": 1.0, "Radius": 1.0, "Age": 4.6, "Spectral_Class": "G"}
        },
        {
            "name": "Gas Giant Planet (Jupiter-like)",
            "data": {"Mass": 0.00095, "Temperature": 128.0, "Luminosity": 1e-9, "Radius": 0.10, "Age": 4.5, "Spectral_Class": "None"}
        },
        {
            "name": "Icy Comet (Halley-like)",
            "data": {"Mass": 1e-14, "Temperature": 180.0, "Luminosity": 1e-20, "Radius": 1e-6, "Age": 4.5, "Spectral_Class": "None"}
        }
    ]
    
    for demo in demos:
        print(f"=== DEMO: {demo['name']} ===")
        process_and_predict(demo["data"], clf, reg, label_encoder)

def main():
    print("==================================================")
    print("  Celestial Object Classification & Regression   ")
    print("==================================================")
    
    # Load ML Assets
    try:
        clf, reg, label_encoder = load_ml_assets()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
        
    # Check if we should automatically run demos (non-interactive shell check)
    is_interactive = sys.stdin.isatty()
    
    if not is_interactive:
        print("Non-interactive terminal detected. Running pre-defined demonstrations.")
        run_demos(clf, reg, label_encoder)
        return
        
    # Interactive selection loop
    print("Choose prediction mode:")
    print("  1. Interactive custom input")
    print("  2. Run pre-defined demonstration cases (Sun, Jupiter, Comet)")
    
    try:
        choice = input("Enter choice (1 or 2) [1]: ").strip()
    except EOFError:
        choice = "2"
        
    if choice == "2":
        run_demos(clf, reg, label_encoder)
    else:
        print("\nPlease enter the physical properties of the celestial object:")
        
        # Guide typical ranges
        print("💡 Guidelines:")
        print("  - Stars: Mass ~ 0.08 to 60, Temp ~ 2500 to 40000, Lum ~ 1e-4 to 1e6, Radius ~ 0.1 to 100")
        print("  - Planets: Mass ~ 1e-5 to 0.015, Temp ~ 50 to 1500, Lum ~ 1e-12 to 1e-7, Radius ~ 0.001 to 0.15")
        print("  - Comets/Asteroids: Mass ~ 1e-17 to 1e-10, Temp ~ 40 to 250, Lum ~ 1e-22 to 1e-17, Radius ~ 1e-7 to 1e-4")
        print()
        
        mass = get_float_input("1. Mass (relative to Solar Masses)", 1.0)
        temp = get_float_input("2. Temperature (Kelvin)", 5800.0)
        lum = get_float_input("3. Luminosity (relative to Solar Luminosity)", 1.0)
        rad = get_float_input("4. Radius (relative to Solar Radii)", 1.0)
        age = get_float_input("5. Age (Billion Years)", 4.6)
        
        allowed_classes = list(SPECTRAL_MAPPING.keys())
        spec = get_string_input(
            f"6. Spectral Class ({', '.join(allowed_classes)})",
            "G",
            allowed_classes
        )
        
        user_features = {
            "Mass": mass,
            "Temperature": temp,
            "Luminosity": lum,
            "Radius": rad,
            "Age": age,
            "Spectral_Class": spec
        }
        
        process_and_predict(user_features, clf, reg, label_encoder)

if __name__ == "__main__":
    main()
