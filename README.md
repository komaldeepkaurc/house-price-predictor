# House Price Prediction using Advanced Regression Techniques

## Project Overview

This project predicts residential house prices using machine learning regression models on the Kaggle House Prices dataset.

The objective was not only to improve prediction accuracy, but also to understand how preprocessing, feature engineering, regularization, and model behavior affect generalization performance on structured tabular data.

---

## Dataset

Dataset: Kaggle - House Prices: Advanced Regression Techniques

- 1460 training samples
- 80 explanatory features
- Mix of numerical and categorical variables
- Target variable: `SalePrice`

The dataset includes:
- property dimensions
- neighborhood information
- basement and garage features
- quality ratings
- construction details

---

## Workflow

### 1. Data Preprocessing
- Missing value handling
- Median/mode imputation
- Domain-aware null handling (`None` categories)
- One-hot encoding
- Feature alignment between train and test sets

### 2. Exploratory Data Analysis
- Distribution analysis
- Correlation analysis
- Residual analysis
- Outlier detection
- Feature-target relationship visualization

### 3. Skewness Handling
Highly skewed numerical features were transformed using log transformation to improve model stability and reduce variance.

Target transformation:

SalePrice → log(1 + SalePrice)

### 4. Feature Engineering
Several custom features were created:

- `TotalSF`
- `HouseAge`
- `TotalBath`
- `TotalPorchSF`
- `Qual_TotalSF`
- `LuxuryScore`
- `BathQual`

Feature engineering significantly improved linear model performance.

---

## Models Used

| Model | Purpose |
|---|---|
| Ridge Regression | Regularized linear baseline |
| Lasso Regression | Feature selection + regularization |
| Random Forest Regressor | Nonlinear ensemble learning |
| Gradient Boosting Regressor | Sequential boosting |
| XGBoost Regressor | Advanced boosting |
| Ensemble Model | Prediction averaging |

---

## Model Performance

| Model | RMSE |
|---|---|
| Lasso Regression | 0.1123 |
| Final Ensemble | 0.1138 |
| Ridge Regression | 0.1172 |
| XGBoost | 0.1254 |
| Gradient Boosting | 0.1282 |
| Random Forest | 0.1415 |

---

## Key Insights

### Regularized Linear Models Performed Best
After preprocessing and feature engineering, the dataset became highly linear in nature. Lasso Regression generalized better than more complex nonlinear models.

### Feature Engineering Was Highly Impactful
Interaction-based features improved predictive performance more effectively than increasing model complexity.

### Complex Models Showed Higher Variance
Random Forest and XGBoost captured nonlinear relationships but generalized less effectively on the relatively small dataset.

### Residual Analysis Revealed Prediction Bias
Expensive houses were frequently underpredicted, indicating regression toward the mean and sparse representation of luxury homes in the dataset.

---

## Final Conclusion

This project demonstrated that:
- preprocessing quality strongly affects model performance
- regularization is highly effective for structured tabular data
- feature engineering can outperform additional model complexity
- residual analysis is important for understanding model behavior

The final pipeline achieved stable and competitive performance while maintaining interpretability and strong generalization.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## Future Improvements

Potential future enhancements:
- LightGBM implementation
- Hyperparameter optimization with Optuna
- SHAP explainability
- Streamlit deployment
- sklearn Pipeline automation

---

## Author

Komaldeep
