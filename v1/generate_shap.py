import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

PRE = "preprocessed"
X_train = pd.read_csv(f"{PRE}/X_train.csv").values.astype(np.float32)
X_test  = pd.read_csv(f"{PRE}/X_test.csv").values.astype(np.float32)
y_train = pd.read_csv(f"{PRE}/y_train.csv").squeeze().values.astype(int)

feature_names = joblib.load(f"{PRE}/feature_names.pkl")
class_weights = joblib.load(f"{PRE}/class_weights.pkl")

X_test_df = pd.DataFrame(X_test, columns=feature_names)

print("Training XGBoost quickly for SHAP generation...")
sample_weights = np.array([class_weights[y] for y in y_train])
model = xgb.XGBClassifier(
    objective='multi:softprob', num_class=3, n_estimators=400, learning_rate=0.05, 
    max_depth=6, tree_method='hist', device='cuda', random_state=42
)
model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_df.iloc[:2000])

plt.figure(figsize=(10, 8))
if isinstance(shap_values, list):
    shap_vals_high = shap_values[2]
elif len(np.array(shap_values).shape) == 3:
    shap_vals_high = shap_values[:, :, 2]
else:
    shap_vals_high = shap_values

shap.summary_plot(shap_vals_high, X_test_df.iloc[:2000], show=False)
plt.title("SHAP Summary Plot (XGBoost - Predicting 'High' Risk)")
plt.tight_layout()
plt.savefig('paper_plots/shap_summary.png', dpi=300)
print("SHAP Plot successfully saved!")
