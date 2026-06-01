# Life-style Based Cardiovascular Disease Detection

This repository contains the codebase for early cardiovascular risk prediction in young patients with Type 2 Diabetes Mellitus (T2DM). The project uses structured clinical, metabolic, and lifestyle data to train and evaluate various machine learning and deep learning models, focusing on TabNet for interpretable sequential attention.

## Project Structure

The repository is organized into data preprocessing, model training notebooks, and evaluation scripts:

- **Data Preprocessing**
  - `preprocessing.ipynb`: The core notebook that handles feature engineering, Z-score normalization, data splitting, and synthetic data generation (SMOTE) to handle class imbalances.
- **Model Implementations**
  - `model_random_forest.ipynb`: Implementation and evaluation of the Random Forest classifier.
  - `model_svm.ipynb`: Implementation and evaluation of the Support Vector Machine (SVM) classifier.
  - `model_xgboost.ipynb`: Implementation and evaluation of the XGBoost classifier.
  - `model_tabnet.ipynb`: Implementation of the baseline TabNet architecture.
  - `better_tabnet.py`: A specialized Python script containing a fine-tuned TabNet model architecture with optimized hyperparameters (increased decision steps, expanded attention dimensions, etc.) specifically designed to maximize performance on this dataset.
- **Evaluation & Visualization**
  - `create_comparison_graphs.py`: Script to aggregate performance metrics (Accuracy, F1-Score, ROC-AUC) across all trained models and generate visual comparisons.
- **Directories**
  - `models/`: Stores saved, trained model weights for inference.
  - `plots/`: Contains generated visualizations, confusion matrices, and ROC curves.
  - `preprocessed/`: Contains the cleaned and processed datasets ready for model consumption.

## Documentation
For more detailed information regarding the individual components of this pipeline, please refer to the files in the `docs/` folder:
- [01 Data Preprocessing](docs/01_Data_Preprocessing.md)
- [02 Model Training](docs/02_Model_Training.md)
- [03 Evaluation and Results](docs/03_Evaluation.md)
