# 03. Evaluation and Results

Evaluation is a critical component of this project. Given the heavily imbalanced nature of the dataset, overall accuracy is not the sole indicator of model performance.

## Metrics Used
- **Accuracy**: The raw percentage of correctly classified instances.
- **F1-Score (Weighted & Macro)**: Provides a balanced measure by considering both precision and recall. Essential for evaluating performance on the minority High-Risk class.
- **ROC-AUC**: Evaluates the discriminative ability of the models across varying classification thresholds.
- **Confusion Matrices**: Visually depicts the classification overlaps between Low, Moderate, and High-Risk predictions.

## Comparison Visualization
The script `create_comparison_graphs.py` serves as the central hub for model comparison. 

1. **Metric Aggregation**
   - It aggregates the final evaluation metrics from Random Forest, SVM, XGBoost, and TabNet.
2. **Visual Outputs**
   - It generates comparative bar charts and graphs highlighting the performance disparities across the models.
   - These visualizations are saved to the `plots/` directory.

## Interpretability
A primary goal of utilizing TabNet is its inherent interpretability.
- The TabNet implementation outputs an **Attention Mask / Feature Importance** ranking.
- This highlights which specific features (e.g., Alcohol Consumption, Smoking History) the model relied on most to make its predictions. This provides clinical transparency, ensuring the deep learning model acts as a trustworthy decision support tool.
