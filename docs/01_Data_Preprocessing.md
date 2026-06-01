# 01. Data Preprocessing

The data preprocessing phase is entirely handled within `preprocessing.ipynb`. 
It consumes raw clinical and lifestyle data (e.g., from the BRFSS survey) and prepares it for multi-class classification (Low Risk, Moderate Risk, High Risk).

## Key Pipeline Steps

1. **Feature Engineering**
   - Combines clinically related variable pairs to capture compound risk effects (e.g., smoking combined with alcohol).
   - Generates composite risk scores representing multiple lifestyle factors into a single indicator (e.g., Healthy Diet Score).
   - Performs categorical encoding to convert nominal and ordinal strings into numeric formats suitable for machine learning algorithms.

2. **Z-Score Normalisation**
   - Numerical variables (like BMI and Blood Pressure) are standardized so they have a mean of 0 and a standard deviation of 1.
   - This ensures no single numerical feature disproportionately affects the training process, which is especially important for distance-based models (SVM) and deep learning architectures (TabNet).

3. **Class Imbalance Handling (SMOTE)**
   - The dataset heavily skews towards the "Low Risk" majority class.
   - Synthetic Minority Over-sampling Technique (SMOTE) is applied to the training set. 
   - SMOTE interpolates between existing minority class samples to create synthetic samples, enabling the models to learn more robust decision boundaries for the underrepresented "High Risk" class.

4. **Data Splitting & Export**
   - The dataset is partitioned into Training, Validation, and Test sets using proportional stratification.
   - The final processed datasets are exported to the `preprocessed/` directory for seamless loading into the downstream modelling scripts.
