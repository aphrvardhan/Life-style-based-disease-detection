# 02. Model Training

This project implements multiple machine learning models to establish baselines and ultimately highlight the advantages of a deep learning-based sequential attention network (TabNet).

## Baseline Machine Learning Models

The following traditional machine learning models are implemented in their respective Jupyter Notebooks. They consume the processed tabular data and output predicted risk categories.

1. **Random Forest (`model_random_forest.ipynb`)**
   - An ensemble learning method leveraging multiple decision trees.
   - Robust against overfitting and serves as a strong traditional baseline for tabular classification.

2. **Support Vector Machine (`model_svm.ipynb`)**
   - Implements a hyperplane-based classification strategy.
   - Relies heavily on the Z-score normalization applied during preprocessing.

3. **XGBoost (`model_xgboost.ipynb`)**
   - An optimized distributed gradient boosting library.
   - Highly performant on structured tabular data, making it a critical benchmark against the deep learning model.

## Deep Learning Models

1. **TabNet Baseline (`model_tabnet.ipynb`)**
   - An initial implementation of the TabNet classifier.
   - Features a sequential attention mechanism that dynamically selects the most relevant features at each decision step, enhancing interpretability without sacrificing predictive power.

2. **Fine-Tuned TabNet (`better_tabnet.py`)**
   - Contains a highly optimized configuration of the TabNet architecture.
   - Key hyperparameter adjustments include:
     - Increased decision steps (T).
     - Expanded attention dimensions (nd/na).
     - Adjusted feature reuse coefficient to penalize redundant feature selection.
   - This script trains the final, most accurate model and handles hyperparameter scheduling (e.g., StepLR) and early stopping to prevent overfitting.
