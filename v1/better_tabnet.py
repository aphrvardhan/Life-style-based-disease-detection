import numpy as np
import pandas as pd
import joblib, os
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("="*70)
    print("pytorch-tabnet is NOT installed.")
    print("Run:  pip install pytorch-tabnet")
    print("="*70)
    exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
PRE  = "preprocessed"
MDIR = "models/tabnet_better"; os.makedirs(MDIR, exist_ok=True)
PDIR = "plots/tabnet_better";  os.makedirs(PDIR, exist_ok=True)
LABEL = {0: 'Low', 1: 'Moderate', 2: 'High'}

# 1. Load preprocessed splits
print("Loading preprocessed data...")
X_train = pd.read_csv(f"{PRE}/X_train.csv").values.astype(np.float32)
X_val   = pd.read_csv(f"{PRE}/X_val.csv").values.astype(np.float32)
X_test  = pd.read_csv(f"{PRE}/X_test.csv").values.astype(np.float32)
y_train = pd.read_csv(f"{PRE}/y_train.csv").squeeze().values.astype(int)
y_val   = pd.read_csv(f"{PRE}/y_val.csv").squeeze().values.astype(int)
y_test  = pd.read_csv(f"{PRE}/y_test.csv").squeeze().values.astype(int)
feature_names = joblib.load(f"{PRE}/feature_names.pkl")

print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# 2. Define Optimized Architecture
print("\nInitializing better TabNet architecture...")
# Increased capacity and disabled uniform sampling weights to prioritize global accuracy.
tabnet = TabNetClassifier(
    n_d             = 64,           # Increased decision step width
    n_a             = 64,           # Increased attention width
    n_steps         = 5,            # Slightly deeper
    gamma           = 1.5,
    n_independent   = 2,
    n_shared        = 2,
    lambda_sparse   = 1e-4,         # Less aggressive sparsity penalty
    optimizer_fn    = __import__('torch').optim.Adam,
    optimizer_params= {'lr': 1e-3, 'weight_decay': 1e-5},
    scheduler_fn    = __import__('torch').optim.lr_scheduler.CosineAnnealingLR,
    scheduler_params= {'T_max': 50, 'eta_min': 1e-5},
    mask_type       = 'entmax',
    input_dim       = X_train.shape[1],
    output_dim      = 3,
    verbose         = 10,
    seed            = 42,
    device_name     = 'cuda' if __import__('torch').cuda.is_available() else 'auto',
)


# 3. Train
print("\n-- Training Better TabNet --")
# We set weights=0 (default) and remove SMOTE to let the model learn the true distribution, optimizing accuracy.
tabnet.fit(
    X_train            = X_train,
    y_train            = y_train,
    eval_set           = [(X_train, y_train), (X_val, y_val)],
    eval_name          = ['train', 'val'],
    eval_metric        = ['accuracy', 'logloss'],
    max_epochs         = 150,
    patience           = 20,
    batch_size         = 1024,      # Larger batch size
    virtual_batch_size = 128,
    weights            = 0,         # Disabled uniform class balancing
    drop_last          = False,
)

print(f"\nBest epoch : {tabnet.best_epoch}")

# 4. Evaluate
print("\n-- Evaluation on Test Set --")
y_pred  = tabnet.predict(X_test)
y_proba = tabnet.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

print(classification_report(y_test, y_pred, target_names=[LABEL[i] for i in sorted(LABEL)]))
print(f"Accuracy : {acc:.5f}")
print(f"ROC-AUC  : {auc:.5f}")

# Feature importance
feat_imp = pd.Series(tabnet.feature_importances_, index=feature_names)
print("\nTop 10 important features:\n", feat_imp.nlargest(10))

# 5. Save model and plots
print("\nSaving models and generating plots...")
tabnet.save_model(f"{MDIR}/better_tabnet")

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
history = tabnet.history

# Loss curve
axes[0].plot(history['train_logloss'], label='Train logloss')
axes[0].plot(history['val_logloss'],   label='Val logloss')
axes[0].set(title='TabNet Training Curve', xlabel='Epoch', ylabel='Logloss')
axes[0].legend()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=[LABEL[i] for i in sorted(LABEL)]).plot(ax=axes[1])
axes[1].set_title('Better TabNet Confusion Matrix')

# Feature importance
feat_imp.nlargest(20).sort_values().plot(kind='barh', ax=axes[2])
axes[2].set_title('Top-20 TabNet Feature Importances')

plt.tight_layout()
plt.savefig(f"{PDIR}/better_tabnet_results.png", dpi=150)
print(f"Plot saved to ./{PDIR}/better_tabnet_results.png")
print(f"✅ Model saved to ./{MDIR}/")
