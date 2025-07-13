
# Intellectra 2025 – Predictive Modeling

Intellectra is a baby milk producer that offers loyalty points for product purchases made by members.  
Based on customer history, transaction data, and product attributes, participants are challenged to build a binary classification model that can predict whether a customer will repurchase a product in the following month.

This competition is part of the INTELLECTRA 2025 event organized by the Graduate Student Association of Statistics and Data Science at IPB University.

---

## Objectives

- Predict the `next_buy` column: whether a customer will make another purchase next month (1 = yes, 0 = no).
- Apply relevant techniques in statistics, analytics, and machine learning.
- Achieve the best possible performance based on the evaluation metric.

---

## Dataset Overview

The project utilizes multiple datasets:

- `train_transaction_data.csv`  
- `train_label_data.csv`  
- `member_data.csv`  
- `product_data.csv`  
- `prodgram_data.csv`  
- `test_transaction_data.csv`  
- `sample_submission.csv`

These datasets are merged and processed to form a unified dataframe with rich feature space.

---

## Feature Engineering

Features were engineered from both raw and derived fields including:

- **Datetime decomposition** (year, month, day, hour, weekday)
- **Customer profile metrics** (e.g., number of children, eldest/youngest child age)
- **Product-level features** (e.g., price per unit, grammage)
- **Transaction indicators** (`IsWeekend`, `Days_Since_Join`, etc.)
- **Nonlinear transforms**: squares, cube roots, log, exponentials using `AutoFeat`

---

## Handling Imbalanced Data

- **SMOTETomek** was used to both over-sample the minority class and clean Tomek links.
- Target class was fully balanced before training.

---

## Best Model Training: LightGBM + Optuna

### Optimization Goal:
- Maximize `Balanced Accuracy = ½ (Sensitivity + Specificity)`

### Tuning Process:
- 5-Fold Stratified Cross-Validation
- Search Space for:
  - `learning_rate`
  - `num_leaves`, `max_depth`, `min_child_samples`
  - `feature_fraction`, `bagging_fraction`
  - `lambda_l1`, `lambda_l2`
  - `scale_pos_weight`

---

## Evaluation Metric

Leaderboard uses **Balanced Accuracy**, which considers both:
- Sensitivity (Recall for Class 1)
- Specificity (Recall for Class 0)

```
Balanced Accuracy = 0.5 * (TP / (TP + FN) + TN / (TN + FP))
```

---

## Result

This LightGBM + Optuna model achieved the **highest score** among all variations tried, including those with:
- Logistic Regression
- Random Forest
- Manual feature sets

### Best Model Evaluation Summary (LGBM + Optuna)
**Train Set Performance:**
- F1 Score: 0.9998
- Balanced Accuracy: 0.9998
- AUC Score: 1.0000
Confusion Matrix:
```
[[76281     0]
 [   31 76250]]
```

**Validation/Test Set Performance:**
- F1 Score: 0.9055
- Balanced Accuracy: 0.9263
- AUC Score: 0.9861
Confusion Matrix:
```
[[21319   193]
 [  645 4014]]
```

>The model shows strong generalization with a high AUC and balanced accuracy, indicating its ability to detect both positive and negative classes effectively >in a balanced manner.

---

## License

This project is licensed under the MIT License.
