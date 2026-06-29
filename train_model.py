"""
train_model.py
--------------------------------------------------------------------------
Machine Learning Pipeline for Celestial Object Classification & Regression.
This script:
  1. Loads the generated celestial objects dataset.
  2. Cleans the data by removing duplicates and missing values.
  3. Conducts feature engineering (ordinal encoding of Spectral Class).
  4. Encodes targets using LabelEncoder.
  5. Splits the dataset into training and testing sets.
  6. Trains a Random Forest Classifier to identify the Object Type.
  7. Trains a Random Forest Regressor to predict the Object's Lifetime.
  8. Evaluates both models using appropriate metrics.
  9. Saves the models, encoders, and evaluation plots.
--------------------------------------------------------------------------
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
import joblib

# Set seaborn style for premium aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16
})

# Define Paths
DATASET_PATH = os.path.join("dataset", "celestial_objects.csv")
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

def load_and_clean_data(path):
    """Loads dataset, logs dimensions, and performs data cleaning."""
    print("=== Step 1 & 2: Loading & Cleaning Dataset ===")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please run dataset generation first.")
        
    df = pd.read_csv(path, keep_default_na=False, na_values=["", "NaN", "null"])
    print(f"Initial dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 1. Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df.drop_duplicates(inplace=True)
        print(f"-> Removed {dup_count} duplicate rows.")
        
    # 2. Check for missing values
    null_count = df.isnull().any(axis=1).sum()
    if null_count > 0:
        df.dropna(inplace=True)
        print(f"-> Removed {null_count} rows with missing values.")
        
    print(f"Cleaned dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df

def engineer_features(df):
    """
    Applies ordinal mapping to Spectral Class.
    Since stars follow a temperature-based sequence (O B A F G K M)
    and non-stars have no class ('None'), we map them to a physical scale.
    """
    print("=== Step 3 & 4: Feature Engineering & Encoding ===")
    
    # We copy to avoid setting-with-copy warnings
    df_engineered = df.copy()
    
    # Ordinal mapping based on stellar temperature progression
    spectral_mapping = {
        "None": 0,
        "M": 1,
        "K": 2,
        "G": 3,
        "F": 4,
        "A": 5,
        "B": 6,
        "O": 7
    }
    
    print("Mapping Spectral_Class to ordinal integers:")
    for k, v in spectral_mapping.items():
        print(f"  {k} -> {v}")
        
    df_engineered["Spectral_Class_Encoded"] = df_engineered["Spectral_Class"].map(spectral_mapping)
    
    # Check if any values failed to map (yielded NaN)
    if df_engineered["Spectral_Class_Encoded"].isnull().any():
        print("Warning: Unmapped spectral classes detected. Filling with 0 ('None').")
        df_engineered["Spectral_Class_Encoded"].fillna(0, inplace=True)
        
    return df_engineered

def train_and_evaluate():
    # 1. Load and clean
    df = load_and_clean_data(DATASET_PATH)
    
    # 2. Feature engineering
    df = engineer_features(df)
    
    # 3. Target Label Encoding for Classification
    label_encoder = LabelEncoder()
    df["Object_Type_Encoded"] = label_encoder.fit_transform(df["Object_Type"])
    
    # Log object type mapping
    print("\nMapping Object_Type to encoded labels:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {idx}")
    print()
    
    # 4. Prepare Features (X) and Targets (y)
    # Features used: Mass, Temperature, Luminosity, Radius, Age, Spectral_Class_Encoded
    feature_cols = ["Mass", "Temperature", "Luminosity", "Radius", "Age", "Spectral_Class_Encoded"]
    X = df[feature_cols]
    y_class = df["Object_Type_Encoded"]
    y_reg = df["Lifetime"]
    
    # 5. Train-Test Split
    # We perform a single split to ensure reproducibility and consistency
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples\n")
    
    # 6. Train RandomForestClassifier
    print("=== Step 5: Training Random Forest Classifier ===")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X_train, y_train_class)
    print("Classifier trained successfully.\n")
    
    # 7. Train RandomForestRegressor
    print("=== Step 6: Training Random Forest Regressor ===")
    reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    reg.fit(X_train, y_train_reg)
    print("Regressor trained successfully.\n")
    
    # 8. Evaluate Classifier
    print("=== Step 7: Evaluating Classifier ===")
    y_pred_class = clf.predict(X_test)
    
    acc = accuracy_score(y_test_class, y_pred_class)
    prec = precision_score(y_test_class, y_pred_class, average='weighted')
    rec = recall_score(y_test_class, y_pred_class, average='weighted')
    f1 = f1_score(y_test_class, y_pred_class, average='weighted')
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (Weighted)")
    print(f"Recall:    {rec:.4f} (Weighted)")
    print(f"F1 Score:  {f1:.4f} (Weighted)\n")
    
    # 9. Evaluate Regressor
    print("=== Step 8: Evaluating Regressor ===")
    y_pred_reg = reg.predict(X_test)
    
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    print(f"Mean Absolute Error (MAE):       {mae:.4f} Billion Years")
    print(f"Mean Squared Error (MSE):        {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.4f} Billion Years")
    print(f"R² Score (R-squared):            {r2:.4f}\n")
    
    # 10. Save Models and Encoders
    print("=== Step 9: Saving Models & Encoders ===")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    clf_path = os.path.join(MODELS_DIR, "classifier.pkl")
    reg_path = os.path.join(MODELS_DIR, "regressor.pkl")
    le_path = os.path.join(MODELS_DIR, "object_encoder.pkl")
    
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    joblib.dump(label_encoder, le_path)
    
    print(f"Saved classifier to {clf_path}")
    print(f"Saved regressor to {reg_path}")
    print(f"Saved label encoder to {le_path}\n")
    
    # 11. Plotting and saving results
    print("=== Step 10: Generating & Saving Visualization Plots ===")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Plot 1: Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_class, y_pred_class)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title("Confusion Matrix - Celestial Object Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_plot_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")
    
    # Plot 2: Feature Importance (Combined)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Classifier Feature Importance
    clf_importances = clf.feature_importances_
    indices_clf = np.argsort(clf_importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices_clf]
    sns.barplot(
        x=clf_importances[indices_clf], y=sorted_features,
        ax=axes[0], palette="viridis", hue=sorted_features, legend=False
    )
    axes[0].set_title("Classifier Feature Importance")
    axes[0].set_xlabel("Relative Importance")
    
    # Regressor Feature Importance
    reg_importances = reg.feature_importances_
    indices_reg = np.argsort(reg_importances)[::-1]
    sorted_features_reg = [feature_cols[i] for i in indices_reg]
    sns.barplot(
        x=reg_importances[indices_reg], y=sorted_features_reg,
        ax=axes[1], palette="magma", hue=sorted_features_reg, legend=False
    )
    axes[1].set_title("Regressor Feature Importance")
    axes[1].set_xlabel("Relative Importance")
    
    plt.suptitle("Random Forest Feature Importances")
    plt.tight_layout()
    fi_plot_path = os.path.join(OUTPUTS_DIR, "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to {fi_plot_path}")
    
    # Plot 3: True vs Predicted Lifetime Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_reg, y=y_pred_reg, alpha=0.7, color="darkviolet", edgecolor="black")
    # Draw standard diagonal line for perfect fit
    max_val = max(max(y_test_reg), max(y_pred_reg))
    min_val = min(min(y_test_reg), min(y_pred_reg))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Predictions")
    
    plt.title(f"Regressor Results: True vs Predicted Lifetime\n(R² = {r2:.3f})")
    plt.xlabel("True Lifetime (Billion Years)")
    plt.ylabel("Predicted Lifetime (Billion Years)")
    plt.legend()
    plt.tight_layout()
    reg_plot_path = os.path.join(OUTPUTS_DIR, "regression_plot.png")
    plt.savefig(reg_plot_path, dpi=150)
    plt.close()
    print(f"Saved regression prediction plot to {reg_plot_path}\n")
    
    print("ML Pipeline run complete! All outputs generated successfully.")

if __name__ == "__main__":
    train_and_evaluate()
