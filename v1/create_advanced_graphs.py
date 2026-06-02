import json
import re
import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi

notebooks = {
    'Random Forest': 'model_random_forest.ipynb',
    'SVM': 'model_svm.ipynb',
    'XGBoost': 'model_xgboost.ipynb',
    'TabNet': 'model_tabnet.ipynb'
}

# --- Parsers ---
def parse_metrics(filepath, model_name):
    if not os.path.exists(filepath): return None
    with open(filepath, 'r', encoding='utf-8') as f: nb = json.load(f)
    
    full_output = ""
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            for output in cell.get('outputs', []):
                if 'text' in output: full_output += "".join(output['text'])
                
    res = {
        'Train Acc': 0.0, 'Test Acc': 0.0, 'ROC-AUC': 0.0,
        'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0,
        'High_Recall': 0.0, 'High_Precision': 0.0, 'High_F1': 0.0,
        'curve_x': [], 'curve_train': [], 'curve_val': []
    }
    
    # Static Metrics
    if model_name == 'Random Forest':
        m = re.findall(r'OOB=(0.\d+)', full_output)
        if m: res['Train Acc'] = float(m[-1])
        
        # Curves
        epochs = re.findall(r'Epoch (\d+)/\d+.*?OOB=(0.\d+).*?val_acc=(0.\d+)', full_output)
        res['curve_x'] = [int(e[0]) for e in epochs]
        res['curve_train'] = [float(e[1]) for e in epochs]
        res['curve_val'] = [float(e[2]) for e in epochs]
        
    elif model_name == 'SVM':
        m = re.findall(r'train_acc=(0.\d+)', full_output)
        if m: res['Train Acc'] = float(m[-1])
        
        epochs = re.findall(r'Epoch (\d+)/\d+.*?train_acc=(0.\d+).*?val_acc=(0.\d+)', full_output)
        res['curve_x'] = [int(e[0]) for e in epochs]
        res['curve_train'] = [float(e[1]) for e in epochs]
        res['curve_val'] = [float(e[2]) for e in epochs]
        
    elif model_name == 'XGBoost':
        m = re.findall(r'validation_0-merror:(0.\d+)', full_output)
        if m: res['Train Acc'] = 1.0 - float(m[-1])
        
        epochs = re.findall(r'\[(\d+)\].*?validation_0-merror:(0.\d+).*?validation_1-merror:(0.\d+)', full_output)
        res['curve_x'] = [int(e[0]) for e in epochs]
        res['curve_train'] = [1.0 - float(e[1]) for e in epochs]
        res['curve_val'] = [1.0 - float(e[2]) for e in epochs]
        
    elif model_name == 'TabNet':
        m = re.findall(r'train_accuracy:\s*(0.\d+)', full_output)
        if m: res['Train Acc'] = float(m[-1])
        
        epochs = re.findall(r'epoch (\d+).*?train_accuracy:\s*(0.\d+).*?val_accuracy:\s*(0.\d+)', full_output)
        res['curve_x'] = [int(e[0]) for e in epochs]
        res['curve_train'] = [float(e[1]) for e in epochs]
        res['curve_val'] = [float(e[2]) for e in epochs]

    m_acc = re.search(r'Fine-tuned Accuracy\s*:\s*(0.\d+)', full_output)
    if not m_acc: m_acc = re.search(r'Accuracy\s*:\s*(0.\d+)', full_output)
    if m_acc: res['Test Acc'] = float(m_acc.group(1))
        
    m_roc = re.search(r'Fine-tuned ROC-AUC\s*:\s*(0.\d+)', full_output)
    if not m_roc: m_roc = re.search(r'ROC-AUC\s*(?:\(OvR\))?\s*:\s*(0.\d+)', full_output)
    if m_roc: res['ROC-AUC'] = float(m_roc.group(1))
        
    m_macro = re.findall(r'macro avg\s+(0.\d+)\s+(0.\d+)\s+(0.\d+)', full_output)
    if m_macro:
        res['Precision'] = float(m_macro[-1][0])
        res['Recall'] = float(m_macro[-1][1])
        res['F1'] = float(m_macro[-1][2])
        
    m_high = re.findall(r'High\s+(0.\d+)\s+(0.\d+)\s+(0.\d+)', full_output)
    if m_high:
        res['High_Precision'] = float(m_high[-1][0])
        res['High_Recall'] = float(m_high[-1][1])
        res['High_F1'] = float(m_high[-1][2])

    return res

metrics = {k: parse_metrics(v, k) for k, v in notebooks.items()}
os.makedirs('plots', exist_ok=True)

# --- 1. Radar Plot ---
def make_radar_plot():
    categories = ['Train Acc', 'Test Acc', 'ROC-AUC', 'Precision', 'Recall', 'F1']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['blue', 'orange', 'green', 'red']
    for i, (m_name, m_data) in enumerate(metrics.items()):
        values = [m_data[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=m_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
        
    plt.xticks(angles[:-1], categories, color='black', size=12)
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.05)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Performance Radar Chart", size=15, y=1.1)
    plt.tight_layout()
    plt.savefig('plots/radar_plot_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- 2. Line Graphs (Training Curves) ---
def make_line_graphs():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ['blue', 'orange', 'green', 'red']
    
    for i, (m_name, m_data) in enumerate(metrics.items()):
        ax = axes[i]
        x = m_data['curve_x']
        train_y = m_data['curve_train']
        val_y = m_data['curve_val']
        
        if len(x) > 0:
            ax.plot(x, train_y, marker='o', label='Train Acc', color=colors[i])
            ax.plot(x, val_y, marker='x', linestyle='--', label='Val Acc', color='grey')
            ax.set_title(f'{m_name} Training Curve')
            ax.set_xlabel('Epochs / Rounds' if m_name != 'Random Forest' else 'Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No curve data available', ha='center', va='center')
            ax.set_title(f'{m_name} Training Curve')
            
    plt.tight_layout()
    plt.savefig('plots/training_curves_comparison.png', dpi=300)
    plt.close()

# --- 3. Minority Class (High Risk) Performance ---
def make_minority_bar_chart():
    labels = list(metrics.keys())
    high_prec = [metrics[m]['High_Precision'] for m in labels]
    high_rec = [metrics[m]['High_Recall'] for m in labels]
    high_f1 = [metrics[m]['High_F1'] for m in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, high_prec, width, label='Precision (High Risk)', color='salmon')
    rects2 = ax.bar(x, high_rec, width, label='Recall (High Risk)', color='crimson')
    rects3 = ax.bar(x + width, high_f1, width, label='F1-Score (High Risk)', color='darkred')

    ax.set_ylabel('Score')
    ax.set_title('Model Performance on Minority Class ("High" Cardiovascular Risk)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('plots/minority_class_comparison.png', dpi=300)
    plt.close()

make_radar_plot()
make_line_graphs()
make_minority_bar_chart()
print("Advanced graphs created successfully in 'plots' folder!")
