import json
import re
import matplotlib.pyplot as plt
import numpy as np
import os

notebooks = {
    'Random Forest': 'model_random_forest.ipynb',
    'SVM': 'model_svm.ipynb',
    'XGBoost': 'model_xgboost.ipynb',
    'TabNet': 'model_tabnet.ipynb'
}

metrics = {}

def extract_from_notebook(filepath, model_name):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Aggregate all output text to find the metrics
    full_output = ""
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            for output in cell.get('outputs', []):
                if 'text' in output:
                    full_output += "".join(output['text'])
    
    res = {
        'Train Accuracy': 0.0,
        'Test Accuracy': 0.0,
        'ROC-AUC': 0.0,
        'Macro Precision': 0.0,
        'Macro Recall': 0.0,
        'Macro F1': 0.0
    }
    
    # Train Accuracy heuristics
    if model_name == 'Random Forest':
        m = re.findall(r'OOB=(0.\d+)', full_output)
        if m: res['Train Accuracy'] = float(m[-1])
    elif model_name == 'SVM':
        m = re.findall(r'train_acc=(0.\d+)', full_output)
        if m: res['Train Accuracy'] = float(m[-1])
    elif model_name == 'XGBoost':
        m = re.findall(r'validation_0-merror:(0.\d+)', full_output)
        if m: res['Train Accuracy'] = 1.0 - float(m[-1])
    elif model_name == 'TabNet':
        m = re.findall(r'train_accuracy:\s*(0.\d+)', full_output)
        if m: res['Train Accuracy'] = float(m[-1])
        
    # Fine-tuned or Test Accuracy
    m_acc = re.search(r'Fine-tuned Accuracy\s*:\s*(0.\d+)', full_output)
    if not m_acc:
        m_acc = re.search(r'Accuracy\s*:\s*(0.\d+)', full_output)
    if m_acc: res['Test Accuracy'] = float(m_acc.group(1))
        
    # ROC-AUC
    m_roc = re.search(r'Fine-tuned ROC-AUC\s*:\s*(0.\d+)', full_output)
    if not m_roc:
        m_roc = re.search(r'ROC-AUC\s*(?:\(OvR\))?\s*:\s*(0.\d+)', full_output)
    if m_roc: res['ROC-AUC'] = float(m_roc.group(1))
        
    # Classification report macro avg
    # macro avg       0.99      0.75      0.80     10978
    # We find all occurrences and take the last one (usually fine-tuned)
    m_macro = re.findall(r'macro avg\s+(0.\d+)\s+(0.\d+)\s+(0.\d+)', full_output)
    if m_macro:
        res['Macro Precision'] = float(m_macro[-1][0])
        res['Macro Recall'] = float(m_macro[-1][1])
        res['Macro F1'] = float(m_macro[-1][2])
        
    return res

for name, path in notebooks.items():
    data = extract_from_notebook(path, name)
    if data:
        metrics[name] = data
        print(f"Extracted metrics for {name}: {data}")

# Plotting
labels = list(metrics.keys())
train_acc = [metrics[m]['Train Accuracy'] for m in labels]
test_acc = [metrics[m]['Test Accuracy'] for m in labels]
roc_auc = [metrics[m]['ROC-AUC'] for m in labels]
recall = [metrics[m]['Macro Recall'] for m in labels]
f1 = [metrics[m]['Macro F1'] for m in labels]
precision = [metrics[m]['Macro Precision'] for m in labels]

x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - 2.5*width, train_acc, width, label='Train Accuracy', color='lightblue')
rects2 = ax.bar(x - 1.5*width, test_acc, width, label='Test Accuracy', color='steelblue')
rects3 = ax.bar(x - 0.5*width, roc_auc, width, label='ROC-AUC', color='lightcoral')
rects4 = ax.bar(x + 0.5*width, precision, width, label='Macro Precision', color='mediumpurple')
rects5 = ax.bar(x + 1.5*width, recall, width, label='Macro Recall', color='mediumseagreen')
rects6 = ax.bar(x + 2.5*width, f1, width, label='Macro F1', color='goldenrod')

ax.set_ylabel('Scores')
ax.set_title('Comparison of Models on Various Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
ax.set_ylim(0, 1.1)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
output_path = 'plots/combined_model_comparison.png'
plt.savefig(output_path, dpi=300)
print(f"\nPlot saved to {output_path}")
