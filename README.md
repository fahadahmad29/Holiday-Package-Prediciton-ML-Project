# 🏝️ Holiday Package Prediction — Wellness Tourism ML Project

Predicting *who* will buy, so marketing stops calling *everyone*.

## 📌 The Problem

**Trips & Travel.Com** sells 5 packages — Basic, Standard, Deluxe, Super Deluxe, King — but last year only **18% of randomly contacted customers** converted. Cold-calling everyone was expensive and inefficient.

Now the company is launching a new **Wellness Tourism Package** and wants to fix the process: use customer data to find *who's actually likely to buy*, instead of dialing at random.

## 🎯 Goal

Build a classification model that predicts `ProdTaken` (1 = will purchase, 0 = won't) from customer demographics and travel behavior — so marketing spend goes toward high-probability leads.

## 📊 Dataset

`Travel.csv` — customer-level data including:

| Feature | Description |
|---|---|
| `Age`, `Gender`, `MaritalStatus`, `Occupation`, `Designation` | Demographics |
| `MonthlyIncome`, `CityTier` | Economic profile |
| `TypeofContact`, `NumberOfFollowups`, `DurationOfPitch`, `PitchSatisfactionScore` | Sales interaction |
| `ProductPitched`, `PreferredPropertyStar`, `NumberOfTrips`, `Passport`, `OwnCar` | Travel behavior |
| `NumberOfPersonVisiting`, `NumberOfChildrenVisiting` | Group size |
| `ProdTaken` | **Target** — purchased or not |

## 🛠️ Approach

1. **EDA** — understand distributions & relationships (`matplotlib`, `seaborn`, `plotly`)
2. **Preprocessing** — `OneHotEncoder` for categoricals, `StandardScaler` for numerics via `ColumnTransformer`
3. **Baseline models** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
4. **Hyperparameter tuning** — `RandomizedSearchCV` on Random Forest
5. **Evaluation** — Accuracy, F1, Precision, Recall, ROC-AUC

## 🏆 Results

**Tuned Random Forest** (`n_estimators=1000`, `max_features=7`) — best performer:

| Metric | Train | Test |
|---|---|---|
| Accuracy | 1.00 | **0.94** |
| F1 Score | 1.00 | 0.93 |
| Precision | 1.00 | 0.98 |
| Recall | 1.00 | 0.70 |
| ROC-AUC | 1.00 | 0.85 |

High precision (0.98) means when the model flags a customer as a likely buyer, it's right almost every time — exactly what's needed to target marketing spend efficiently.

📈 ROC curve: [`auc.png`](./auc.png)

## 📁 Repo Structure

```
├── Travel.csv              # Dataset
├── randomclassifier.ipynb  # Full analysis: EDA → preprocessing → modeling → tuning
└── auc.png                 # ROC-AUC curve
```

## 🚀 Run It

```bash
git clone https://github.com/fahadahmad29/Holiday-Package-Prediciton-ML-Project.git
cd Holiday-Package-Prediciton-ML-Project
pip install pandas numpy matplotlib seaborn plotly scikit-learn
jupyter notebook randomclassifier.ipynb
```

## 💡 Business Impact

Instead of contacting customers at random (18% hit rate), the company can now **rank leads by predicted purchase probability** and focus marketing calls on the top segment — cutting wasted outreach and lifting conversion for the new Wellness Tourism Package.

## 👤 Author

**Fahad Ahmad Khan**
