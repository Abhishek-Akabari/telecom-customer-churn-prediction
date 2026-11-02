# Telecom Customer Churn Prediction

Predicting which telecom customers are likely to leave, using regularised logistic regression and an autoencoder-based neural network. This was a group project for the Predictive Data Analysis module of my MSc in Artificial Intelligence at Brunel University London. **This repository contains the two models I built and evaluated myself**: the logistic regression and the autoencoder + classifier.

![ROC comparison](images/fig_comparison_roc.png)

## The problem

The dataset has 7,043 customers and 20 columns covering contract type, services, charges, tenure and demographics. 26.5% of customers churned. With that imbalance, a model that predicts "stays" for everyone is right about 73% of the time and still useless, so I focused on AUC-ROC, F1 and precision-recall rather than accuracy alone.

## What I did

**Data preparation**
- Clamped 120 negative monthly charges to zero.
- Filled missing service values based on context. For example, a missing "Online Security" for a customer without internet means "No Internet", not unknown.
- Engineered features: a service usage score (how many add-on services a customer has), revenue per tenure month, a senior flag, and a contract × tenure interaction.
- Encoded the categorical fields and dropped `Total Charges`, which is almost a product of monthly charge and tenure.
- Used an 80/20 stratified train/test split with `random_state=42`.

**Model 1: Logistic regression (L1 / L2 / ElasticNet)**
- Scaled the features and applied SMOTE to the training data only, so no synthetic rows leak into the test set.
- Ran a grid search over penalty type and regularisation strength with 5-fold stratified cross-validation, scored on AUC-ROC.
- Turned the coefficients into odds ratios to explain what drives churn.

**Model 2: Autoencoder + classifier head (TensorFlow/Keras)**
- Pre-trained an autoencoder (34 → 64 → 32 → 16 → 32 → 64 → 34) on the training features without labels.
- Froze the encoder and trained a small classifier head on top, using class weights and focal loss to handle the imbalance, then fine-tuned end to end with a low learning rate.
- Visualised the 16-dimensional bottleneck with t-SNE.

## Results (test set, 1,409 customers)

| Model | AUC-ROC | F1 (churn) | Accuracy | Avg precision |
|---|---|---|---|---|
| Logistic regression | 0.873 | 0.667 | 0.809 | 0.727 |
| Autoencoder + classifier | 0.869 | 0.670 | 0.803 | 0.723 |
| Weighted ensemble (LR + AE) | 0.874 | 0.669 | 0.820 | 0.728 |

These figures come from the most recent run of the notebook. Neural network training isn't perfectly deterministic, so a re-run can shift them slightly.

The simpler logistic regression did about as well as the deep learning model, and it's much easier to explain to a business.

## What drives churn

From the logistic regression odds ratios:

- **Raise churn risk:** month-to-month contracts (about 3.5× the odds), higher monthly charges, and being a senior customer.
- **Lower churn risk:** two-year contracts, having dependents, and longer tenure on a month-to-month contract.

![Logistic regression coefficients](images/fig_lr_coefficients.png)

## Limitations

- The decision threshold and ensemble weights were tuned on the test set, so the "tuned" scores are a little optimistic. With more time I'd tune them on a separate validation split.
- This is one telecom provider's data, so the drivers may not carry over to other markets.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook churn_models.ipynb
```

Keep `telecom_customer_churn_20_columns.xlsx` in the same folder as the notebook. Figures and CSV outputs are written to `outputs/`.

## Tools

Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), TensorFlow/Keras, matplotlib, seaborn.

## Data

A 20-column version of the publicly available Telecom Customer Churn dataset (7,043 customers), prepared by our project group for the module.

---
Abhishek Akabari · [LinkedIn](https://www.linkedin.com/in/abhishek-akabari-1ba63626b) · [GitHub](https://github.com/Abhishek-Akabari)
