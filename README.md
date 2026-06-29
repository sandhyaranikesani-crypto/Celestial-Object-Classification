# Celestial Object Classification & Lifetime Regression


# 📸 Project Screenshots

## Training Output

![Training](screenshots/train1,2.png)

## Prediction Output

![Prediction](screenshots/predict1,2.png)

## Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)

## Feature Importance

![Feature Importance](outputs/feature_importance.png)

## Regression Plot

![Regression Plot](outputs/regression_plot.png)

A robust, modular, and beginner-friendly Machine Learning project designed to classify celestial objects (Stars, Planets, Moons, Asteroids, and Comets) based on their physical properties and predict their estimated lifetimes using Random Forest models. This project features a complete end-to-end data cleaning, feature engineering, training, and interactive prediction pipeline.

---

## 📂 Project Structure

```text
Celestial-Object-Classification/
│
├── dataset/
│   └── celestial_objects.csv       # Synthetic dataset containing 300+ physical object profiles
│
├── models/
│   ├── classifier.pkl             # Trained Random Forest Classifier model
│   ├── regressor.pkl              # Trained Random Forest Regressor model
│   └── object_encoder.pkl         # Label encoder mapping object categories
│
├── outputs/
│   ├── confusion_matrix.png       # Classification performance visualization
│   ├── feature_importance.png     # Feature importance plots for classifier and regressor
│   └── regression_plot.png        # Regressor actual vs predicted scatter plot
│
├── train_model.py                 # Core model training, data cleaning & evaluation pipeline
├── predict.py                     # Interactive CLI inference & prediction demonstration
├── requirements.txt               # Dependencies list
├── README.md                      # Project documentation
├── LICENSE                        # MIT License
└── .gitignore                     # Standard python ignores (omits model weights, temporary files)
```

---

## 📊 Dataset Description

The project relies on a realistically simulated dataset of **365 celestial bodies** containing physical astronomical properties. To demonstrate data cleaning, the dataset is seeded with duplicate records and empty/missing parameters.

### Features
* **`Mass`**: Mass relative to the Sun (Solar Masses, \(M_{\odot}\)).
* **`Temperature`**: Effective surface temperature in Kelvin (K).
* **`Luminosity`**: Total energy emitted relative to the Sun (Solar Luminosity, \(L_{\odot}\)).
* **`Radius`**: Radius relative to the Sun (Solar Radii, \(R_{\odot}\)).
* **`Age`**: Current estimated age in Billion Years (Gyr).
* **`Spectral_Class`**: Stellar classification sequence (`O`, `B`, `A`, `F`, `G`, `K`, `M`). Non-stellar bodies are designated as `None`.

### Targets
1. **`Object_Type`** (Classification): `Star`, `Planet`, `Moon`, `Asteroid`, `Comet`
2. **`Lifetime`** (Regression): Estimated total lifetime in Billion Years (Gyr).

---

## ⚙️ Machine Learning Workflow

1. **Load Dataset**: Reads CSV data handling custom null boundaries (treating `None` as a valid category for non-stars while parsing empty spaces as `NaN`).
2. **Data Cleaning**:
   * Drops duplicated rows (`df.drop_duplicates()`).
   * Drops rows with missing feature values (`df.dropna()`).
3. **Feature Engineering**: Standardizes the categorical `Spectral_Class` feature using an **ordinal integer mapping** that matches the physical temperature sequence of the Morgan–Keenan spectral types:
   $$\text{None} (0) \to M (1) \to K (2) \to G (3) \to F (4) \to A (5) \to B (6) \to O (7)$$
4. **Label Encoding**: Encodes targets (`Object_Type`) into numeric categories for classifier compatibility.
5. **Train-Test Split**: Divides clean dataset records into an 80/20 train-test ratio for unbiased validation.
6. **Random Forest Classifier**: Fits a classifier to map engineered features to the target class labels.
7. **Random Forest Regressor**: Fits a regressor to map the features to target lifetimes.
8. **Model Evaluation**: Generates metrics (Accuracy, F1, MAE, R² Score) and plots.
9. **Save Models**: Serializes models and label mappings to the `models/` directory using `joblib`.

---

## 🤖 Algorithms Used

* **Random Forest Classifier**: An ensemble learning method using bagging on decision trees to perform robust multi-class classification. It handles highly non-linear decision boundaries between astronomical bodies (like stars vs. asteroids).
* **Random Forest Regressor**: Trains decision tree estimators to predict continuous variables (`Lifetime`). Since celestial body lifetimes are strongly non-linear (e.g., stellar lifetimes scale inversely with \(M^{-1.5}\) or \(M^{-2.5}\)), decision trees capture these relationships without needing complex transformations.

---

## 📈 Results & Evaluation

After cleaning, the final training dataset size consists of **269 samples** with **68 samples** reserved for evaluation.

### 1. Classification Metrics (Object Type)
* **Accuracy**: `1.0000` (100.0%)
* **Precision**: `1.0000` (Weighted)
* **Recall**: `1.0000` (Weighted)
* **F1 Score**: `1.0000` (Weighted)

*Classification is highly accurate because celestial classes occupy distinct clusters separated by multiple orders of magnitude (e.g. mass difference between a star and a comet).*

### 2. Regression Metrics (Lifetime prediction)
* **Mean Absolute Error (MAE)**: `1.7830` Billion Years
* **Root Mean Squared Error (RMSE)**: `2.6221` Billion Years
* **R² Score (R-squared)**: `0.7200` (72% variance explained)

### Visualizations (Saved in `outputs/`)
All figures are saved automatically at the end of `python train_model.py`:
1. `confusion_matrix.png`: Heatmap demonstrating perfect classification across all 5 classes.
2. `feature_importance.png`: Bar chart highlighting which physical features (e.g., `Mass`, `Temperature`) influenced the predictions the most.
3. `regression_plot.png`: Scatter plot comparing the ground truth values against regressor predictions with a diagonal \(y=x\) guide.

---

## 🚀 Installation & Setup

1. **Clone or Open this workspace**:
   Make sure you are in the project folder `Celestial-Object-Classification/`.

2. **Set up a Virtual Environment (Optional but recommended)**:
   ```bash
   python -m venv venv
   # Activate on Windows:
   .\venv\Scripts\activate
   # Activate on macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 How to Run

### Step 1: Train Models & Generate Plots
Run the training pipeline to clean the dataset, train classification/regression models, evaluate them, and output plots:
```bash
python train_model.py
```

### Step 2: Run Predictions
Test predictions using the interactive CLI. When run in a command line window, it will prompt you to choose between entering custom values or running pre-defined demonstrations:
```bash
python predict.py
```

*Example CLI Demo Output:*
```text
=== DEMO: Stellar Body (G-type Star like the Sun) ===
==================================================
                PREDICTION RESULTS                
==================================================
Inputs:
  - Mass:        1.0 Solar Masses
  - Temperature: 5778.0 K
  - Luminosity:  1.0 Solar Luminosity
  - Radius:      1.0 Solar Radii
  - Age:         4.6 Billion Years
  - Spectral Cl: G (Encoded as 3)
--------------------------------------------------
Predicted Type:  Star
Confidence:      95.00%
Est. Lifetime:   9.6953 Billion Years
==================================================
```

---

## 🔮 Future Scope
* **Deep Learning Model**: Implement a multi-output Artificial Neural Network (ANN) using PyTorch to solve classification and regression tasks in a single model.
* **Hyperparameter Tuning**: Run grid search (`GridSearchCV`) to optimize the number of estimators, max depth, and minimum leaf samples.
* **Add Noise Features**: Introduce unrelated features (like "Stellar Coordinate") to observe and prune features with low feature importance.
