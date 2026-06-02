import nbformat as nbf

nb = nbf.v4.new_notebook()

def add_md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def add_code(text):
    nb.cells.append(nbf.v4.new_code_cell(text))

add_md("""# Comprehensive Model Evaluation & Visualization
This notebook trains, evaluates, and compares 5 models for Cardiovascular Risk Prediction:
1. Random Forest (CPU - utilizing all cores)
2. Support Vector Machine (SGD SVC) (CPU)
3. XGBoost (GPU Accelerated)
4. Baseline TabNet (GPU Accelerated)
5. Optimized 'Better' TabNet (GPU Accelerated)

It also generates a massive suite of publication-ready visualizations in the `paper_plots/` directory.
""")

add_code("""
!py -3.13 -m pip install xgboost pytorch-tabnet shap scikit-posthocs matplotlib seaborn scikit-learn pandas numpy
""")

add_code("""
import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.augmentations import ClassificationSMOTE

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report, 
                             confusion_matrix, ConfusionMatrixDisplay, 
                             precision_recall_curve, average_precision_score,
                             roc_curve, auc, brier_score_loss)
from sklearn.calibration import calibration_curve

import shap
import scikit_posthocs as sp

warnings = __import__('warnings')
warnings.filterwarnings('ignore')

os.makedirs('paper_plots', exist_ok=True)
os.makedirs('models/comprehensive', exist_ok=True)
PRE = "preprocessed"
LABEL = {0: 'Low', 1: 'Moderate', 2: 'High'}

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
""")

add_md("## 1. Load Data")
add_code("""
print("Loading preprocessed data...")
X_train = pd.read_csv(f"{PRE}/X_train.csv").values.astype(np.float32)
X_val   = pd.read_csv(f"{PRE}/X_val.csv").values.astype(np.float32)
X_test  = pd.read_csv(f"{PRE}/X_test.csv").values.astype(np.float32)

y_train = pd.read_csv(f"{PRE}/y_train.csv").squeeze().values.astype(int)
y_val   = pd.read_csv(f"{PRE}/y_val.csv").squeeze().values.astype(int)
y_test  = pd.read_csv(f"{PRE}/y_test.csv").squeeze().values.astype(int)

class_weights = joblib.load(f"{PRE}/class_weights.pkl")
feature_names = joblib.load(f"{PRE}/feature_names.pkl")

X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")
""")

add_md("""## 2. Define Models
Here we define the 5 models.
### TabNet Baseline vs Better TabNet Differences:
- **Baseline TabNet**: Uses smaller capacity (`n_d=32`, `n_a=32`), `ClassificationSMOTE` for augmentation, and standard class weighting.
- **Better TabNet**: Increases capacity (`n_d=64`, `n_a=64`), removes SMOTE to learn the true underlying distribution, reduces sparsity penalty (`lambda_sparse=1e-4`), and uses a larger batch size (1024) to optimize for global accuracy rather than forced balance.
""")

add_code("""
models = {}

# 1. Random Forest (CPU)
models['Random Forest'] = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_split=5, class_weight=class_weights, n_jobs=-1, random_state=42
)

# 2. SVM (CPU - SGD)
models['SVM'] = SGDClassifier(
    loss='modified_huber', penalty='l2', alpha=0.001, class_weight=class_weights, max_iter=2000, n_jobs=-1, random_state=42
)

# 3. XGBoost (GPU)
sample_weights = np.array([class_weights[y] for y in y_train])
models['XGBoost'] = xgb.XGBClassifier(
    objective='multi:softprob', num_class=3, n_estimators=400, learning_rate=0.05, 
    max_depth=6, tree_method='hist', device='cuda', random_state=42
)

# 4. TabNet Baseline (GPU)
models['TabNet Baseline'] = TabNetClassifier(
    n_d=32, n_a=32, n_steps=4, gamma=1.3,
    optimizer_fn=torch.optim.Adam, optimizer_params={'lr': 2e-3},
    mask_type='sparsemax', seed=42, device_name='cuda'
)

# 5. Better TabNet (GPU)
models['Better TabNet'] = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5, gamma=1.5, lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam, optimizer_params={'lr': 1e-3},
    mask_type='entmax', seed=42, device_name='cuda'
)
""")

add_md("## 3. Train Models and Store Histories")

add_code("""
histories = {}
trained_models = {}
metrics_res = {}
cv_scores = {}

# We will run 3-Fold CV for the Critical Difference Diagram
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

from sklearn.base import clone

for name, model in models.items():
    print(f"\\n{'='*40}\\nTraining {name}...\\n{'='*40}")
    t0 = time.time()
    
    # Custom Cross Validation (Critical Difference)
    print("Running 3-Fold CV...")
    fold_accs = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]
        
        # Clone model to avoid state leakage
        cv_model = clone(model)
        
        if 'TabNet' in name:
            if name == 'TabNet Baseline':
                cv_model.fit(
                    X_tr, y_tr, eval_set=[(X_v, y_v)],
                    eval_metric=['accuracy'], max_epochs=80, patience=15, 
                    batch_size=512, augmentations=ClassificationSMOTE(p=0.2)
                )
            else:
                cv_model.fit(
                    X_tr, y_tr, eval_set=[(X_v, y_v)],
                    eval_metric=['accuracy'], max_epochs=120, patience=20, 
                    batch_size=1024, weights=0
                )
            y_pred_val = cv_model.predict(X_v)
        elif name == 'XGBoost':
            sample_weights_cv = np.array([class_weights[y] for y in y_tr])
            cv_model.fit(X_tr, y_tr, sample_weight=sample_weights_cv, eval_set=[(X_v, y_v)], verbose=0)
            y_pred_val = cv_model.predict(X_v)
        else:
            cv_model.fit(X_tr, y_tr)
            y_pred_val = cv_model.predict(X_v)
            
        fold_accs.append(accuracy_score(y_v, y_pred_val))
        
    cv_scores[name] = np.array(fold_accs)
    print(f"CV Accuracies for {name}: {cv_scores[name]}")
    
    # Main Training on Full Train Set
    print("\\nTraining on Full Training Set...")
    if 'TabNet' in name:
        if name == 'TabNet Baseline':
            model.fit(
                X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_name=['train', 'val'], eval_metric=['accuracy', 'logloss'],
                max_epochs=80, patience=15, batch_size=512, augmentations=ClassificationSMOTE(p=0.2)
            )
        else:
            model.fit(
                X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_name=['train', 'val'], eval_metric=['accuracy', 'logloss'],
                max_epochs=120, patience=20, batch_size=1024, weights=0
            )
        histories[name] = model.history
    
    elif name == 'XGBoost':
        model.fit(
            X_train, y_train, sample_weight=sample_weights,
            eval_set=[(X_train, y_train), (X_val, y_val)], verbose=50
        )
        res = model.evals_result()
        histories[name] = {'train_logloss': res['validation_0']['mlogloss'], 'val_logloss': res['validation_1']['mlogloss']}
    
    else:
        model.fit(X_train, y_train)
        
    trained_models[name] = model
    
    # Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True)
    auc_val = roc_auc_score(y_test, y_proba, multi_class='ovr') if y_proba is not None else 0.0
    
    metrics_res[name] = {
        'Test Acc': acc,
        'ROC-AUC': auc_val,
        'Precision': rep['macro avg']['precision'],
        'Recall': rep['macro avg']['recall'],
        'F1': rep['macro avg']['f1-score'],
        'High_Recall': rep['2']['recall'],
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    print(f"\\n{name} Test Accuracy: {acc:.4f} | ROC-AUC: {auc_val:.4f}")
    print(f"Time taken: {time.time()-t0:.1f}s")
""")

add_md("## 4. Visualizations (Filling the Paper)")

add_code("""
# 1. Overlayed ROC Curves
plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green', 'red', 'purple']
for i, (name, m_data) in enumerate(metrics_res.items()):
    if m_data['y_proba'] is not None:
        # Binarize for 'High' Risk (class 2)
        y_test_bin = (y_test == 2).astype(int)
        fpr, tpr, _ = roc_curve(y_test_bin, m_data['y_proba'][:, 2])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Target: High Risk)')
plt.legend(loc="lower right")
plt.savefig('paper_plots/roc_curves.png', dpi=300)
plt.show()
""")

add_code("""
# 2. Precision-Recall Curves
plt.figure(figsize=(10, 8))
for i, (name, m_data) in enumerate(metrics_res.items()):
    if m_data['y_proba'] is not None:
        y_test_bin = (y_test == 2).astype(int)
        precision, recall, _ = precision_recall_curve(y_test_bin, m_data['y_proba'][:, 2])
        ap = average_precision_score(y_test_bin, m_data['y_proba'][:, 2])
        plt.plot(recall, precision, color=colors[i], lw=2, label=f'{name} (AP = {ap:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Target: High Risk)')
plt.legend(loc="upper right")
plt.savefig('paper_plots/pr_curves.png', dpi=300)
plt.show()
""")

add_code("""
# 3. Calibration Curves (Replacement for Survival Curves)
plt.figure(figsize=(10, 8))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for i, (name, m_data) in enumerate(metrics_res.items()):
    if m_data['y_proba'] is not None:
        y_test_bin = (y_test == 2).astype(int)
        prob_pos = m_data['y_proba'][:, 2]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test_bin, prob_pos, n_bins=10)
        
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", color=colors[i], label=name)
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2, color=colors[i])

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration Plots (Reliability Curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.savefig('paper_plots/calibration_curves.png', dpi=300)
plt.show()
""")

add_code("""
# 4. Critical Difference Diagram (Nemenyi Test)
# Using the 3-Fold CV scores we collected
cv_df = pd.DataFrame(cv_scores)

# Rank the methods
ranks = cv_df.rank(axis=1, ascending=False)
mean_ranks = ranks.mean()

plt.figure(figsize=(10, 3))
sp.sign_plot(sp.posthoc_nemenyi_friedman(cv_df.values))
plt.title("Nemenyi Post-Hoc Test P-Values Heatmap")
plt.savefig('paper_plots/nemenyi_heatmap.png', dpi=300)
plt.show()

print("Mean Ranks (Lower is better):")
print(mean_ranks.sort_values())
""")

add_code("""
# 5. Radar Plot
categories = ['Test Acc', 'ROC-AUC', 'Precision', 'Recall', 'F1', 'High_Recall']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i, (name, m_data) in enumerate(metrics_res.items()):
    values = [m_data[c] for c in categories]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=colors[i])
    ax.fill(angles, values, alpha=0.1, color=colors[i])

plt.xticks(angles[:-1], categories, color='black', size=12)
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
plt.ylim(0, 1.05)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title("Model Performance Radar Chart", size=15, y=1.1)
plt.tight_layout()
plt.savefig('paper_plots/radar_plot.png', dpi=300, bbox_inches='tight')
plt.show()
""")

add_code("""
# 6. Simplified Confusion Matrices
fig, axes = plt.subplots(1, 5, figsize=(25, 4))

for i, (name, m_data) in enumerate(metrics_res.items()):
    cm = confusion_matrix(y_test, m_data['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Low', 'Mod', 'High'])
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(name)

plt.tight_layout()
plt.savefig('paper_plots/confusion_matrices.png', dpi=300)
plt.show()
""")

add_code("""
# 7. Training Line Graphs
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

if 'XGBoost' in histories:
    axes[0].plot(histories['XGBoost']['train_logloss'], label='Train Logloss')
    axes[0].plot(histories['XGBoost']['val_logloss'], label='Val Logloss')
    axes[0].set_title('XGBoost Training Curve')
    axes[0].set_xlabel('Rounds')
    axes[0].legend()

if 'TabNet Baseline' in histories:
    axes[1].plot(histories['TabNet Baseline']['train_logloss'], label='Train')
    axes[1].plot(histories['TabNet Baseline']['val_logloss'], label='Val')
    axes[1].set_title('TabNet Baseline')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()

if 'Better TabNet' in histories:
    axes[2].plot(histories['Better TabNet']['train_logloss'], label='Train')
    axes[2].plot(histories['Better TabNet']['val_logloss'], label='Val')
    axes[2].set_title('Better TabNet')
    axes[2].set_xlabel('Epochs')
    axes[2].legend()

plt.tight_layout()
plt.savefig('paper_plots/training_curves.png', dpi=300)
plt.show()
""")

add_code("""
# 8. Feature Importance Heatmap
fi_df = pd.DataFrame(index=feature_names)

if hasattr(trained_models['Random Forest'], 'feature_importances_'):
    fi_df['Random Forest'] = trained_models['Random Forest'].feature_importances_
    
if hasattr(trained_models['XGBoost'], 'feature_importances_'):
    fi_df['XGBoost'] = trained_models['XGBoost'].feature_importances_
    
if hasattr(trained_models['TabNet Baseline'], 'feature_importances_'):
    fi_df['TabNet Baseline'] = trained_models['TabNet Baseline'].feature_importances_
    
if hasattr(trained_models['Better TabNet'], 'feature_importances_'):
    fi_df['Better TabNet'] = trained_models['Better TabNet'].feature_importances_

# Normalize to max 1 for fair color scaling
fi_df = fi_df / fi_df.max()
top_features = fi_df.mean(axis=1).nlargest(15).index
fi_df_top = fi_df.loc[top_features]

plt.figure(figsize=(10, 8))
sns.heatmap(fi_df_top, annot=True, cmap='YlOrRd', fmt=".2f")
plt.title('Normalized Feature Importance Heatmap (Top 15)')
plt.tight_layout()
plt.savefig('paper_plots/feature_importance_heatmap.png', dpi=300)
plt.show()
""")

add_code("""
# 9. SHAP Values (for XGBoost)
print("Calculating SHAP values for XGBoost...")
explainer = shap.TreeExplainer(trained_models['XGBoost'])
# Use a subset to save time
shap_values = explainer.shap_values(X_test_df.iloc[:2000])

plt.figure(figsize=(10, 8))
# Multi-class SHAP outputs a 3D array for XGBoost in modern SHAP versions (samples, features, classes)
# We plot class 2 (High Risk)
if isinstance(shap_values, list):
    shap_vals_high = shap_values[2]
elif len(shap_values.shape) == 3:
    shap_vals_high = shap_values[:, :, 2]
else:
    shap_vals_high = shap_values

shap.summary_plot(shap_vals_high, X_test_df.iloc[:2000], show=False)
plt.title("SHAP Summary Plot (XGBoost - Predicting 'High' Risk)")
plt.tight_layout()
plt.savefig('paper_plots/shap_summary.png', dpi=300)
plt.show()
print("All graphs successfully generated and saved to 'paper_plots/' directory!")
""")

with open('comprehensive_model_comparison.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
