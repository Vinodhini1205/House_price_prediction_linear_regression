# House Price Prediction

A machine learning project that predicts median house values in California using Linear Regression, built on the California Housing dataset from scikit-learn.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
</div>

---

## Overview

Predicting house prices is one of the most common regression problems in machine learning, with real applications in real estate valuation, mortgage lending, and urban planning. This project builds a **Linear Regression model** to predict the median house value for California districts based on features such as income, house age, room counts, population, and location.

The workflow covers the full modeling pipeline: loading the data, exploring it, training a model, and evaluating its performance with standard regression metrics.

### Objectives
- Load and explore the California Housing dataset
- Perform exploratory data analysis, including a correlation heatmap
- Split the data into training and test sets
- Train a Linear Regression model to predict median house value
- Evaluate the model using MAE, RMSE, and R-squared
- Visualize predicted values against actual values

---

## Dataset

This project uses the **California Housing dataset**, loaded directly via `sklearn.datasets.fetch_california_housing`. It is derived from the 1990 U.S. Census and contains 20,640 records across 8 features plus the target variable.

| Feature | Description |
|---|---|
| `MedInc` | Median income in the block group |
| `HouseAge` | Median house age in the block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| `MedHouseVal` | Median house value (target variable, in $100,000s) |

The dataset has no missing values, so no imputation was required.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `matplotlib` / `seaborn` | Data visualization (correlation heatmap, prediction plot) |
| `scikit-learn` | Dataset loading, train/test split, Linear Regression model, evaluation metrics |

---

## Exploratory Data Analysis

- Checked dataset structure and data types with `.info()`
- Confirmed there are no missing values across all columns
- Reviewed summary statistics (mean, std, min, max, quartiles) for every feature
- Plotted a **correlation heatmap** to examine relationships between features and the target variable, highlighting which features are most associated with house value (notably `MedInc`)

---

## Model Training

**Approach:** Linear Regression

1. Split the data into features (`X`) and target (`y = MedHouseVal`)
2. Performed an 80/20 train-test split (`random_state=42` for reproducibility)
3. Trained a `LinearRegression` model on the training set
4. Generated predictions on the held-out test set

```python
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## Model Evaluation

The model was evaluated on the test set using three standard regression metrics:

| Metric | Value |
|---|---|
| MAE (Mean Absolute Error) | 0.533 |
| RMSE (Root Mean Squared Error) | 0.746 |
| R-squared Score | 0.576 |

An **Actual vs Predicted** scatter plot was also generated to visually assess how closely the model's predictions track the true median house values.

---

## Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Project
1. Clone this repository
   ```bash
   git clone https://github.com/<your-username>/house-price-prediction.git
   cd house-price-prediction
   ```
2. Open and run the notebook
   ```bash
   jupyter notebook House_price_prediction.ipynb
   ```

No external dataset download is required — the data is fetched directly through scikit-learn.

---

## Project Structure

```
house-price-prediction/
|
|-- House_price_prediction.ipynb   # Main notebook (EDA + Linear Regression model)
|-- README.md                      # Project documentation
```

---

## Future Improvements

- Engineer additional features (e.g. rooms per household, population density)
- Try more powerful models such as Random Forest, Gradient Boosting, or XGBoost
- Apply feature scaling and regularization (Ridge, Lasso) to compare performance
- Perform cross-validation and hyperparameter tuning
- Deploy the model as a simple web app for interactive price predictions

---

## Acknowledgements

- [scikit-learn](https://scikit-learn.org/) for the California Housing dataset and modeling tools
