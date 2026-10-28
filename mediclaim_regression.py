# %% [markdown]
# # Mediclaim: Linear, Ridge and Lasso Regression
# %% Imports and data
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

URL = "https://itv-contentbucket.s3.ap-south-1.amazonaws.com/Exams/ML/Linear+Regression/mediclaim.csv"
df = pd.read_csv(URL)          # or pd.read_csv("mediclaim.csv") if downloaded
print(df.shape); display(df.head()) if "display" in globals() else print(df.head())
print(df.info()); print(df.describe())

# %% 1. FEATURE ENGINEERING
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# Encode categorical variables
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=int)
print(df.head())

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap"); plt.show()

# %% 2. EXTRACT X AND y
X = df.drop("charges", axis=1)
y = df["charges"]
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (needed so the Ridge/Lasso penalty treats all features fairly)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# %% Helper to evaluate any model
def evaluate(name, model, Xtr, Xte):
    p_tr, p_te = model.predict(Xtr), model.predict(Xte)
    n, k = Xte.shape[0], Xte.shape[1]
    r2 = r2_score(y_test, p_te)
    return {"Model": name,
            "Train R2": r2_score(y_train, p_tr),
            "Test R2": r2,
            "Adj R2": 1 - (1 - r2) * (n - 1) / (n - k - 1),
            "MAE": mean_absolute_error(y_test, p_te),
            "MSE": mean_squared_error(y_test, p_te),
            "RMSE": np.sqrt(mean_squared_error(y_test, p_te))}

# %% 3. LINEAR REGRESSION
lr = LinearRegression().fit(X_train_s, y_train)
print("Intercept:", lr.intercept_)
print(pd.Series(lr.coef_, index=X.columns).sort_values())

# %% 4. EVALUATE LINEAR REGRESSION
res_lr = evaluate("Linear Regression", lr, X_train_s, X_test_s)
print(pd.Series(res_lr))

plt.scatter(y_test, lr.predict(X_test_s), alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual charges"); plt.ylabel("Predicted charges"); plt.title("Linear Regression: actual vs predicted"); plt.show()

# %% 5. RIDGE AND LASSO
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(X_train_s, y_train)
lasso_cv = LassoCV(alphas=np.logspace(-3, 3, 50), cv=5, max_iter=50000).fit(X_train_s, y_train)
print("Best Ridge alpha:", ridge_cv.alpha_, "| Best Lasso alpha:", lasso_cv.alpha_)

ridge = Ridge(alpha=ridge_cv.alpha_).fit(X_train_s, y_train)
lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=50000).fit(X_train_s, y_train)

coefs = pd.DataFrame({"Linear": lr.coef_, "Ridge": ridge.coef_, "Lasso": lasso.coef_}, index=X.columns)
print(coefs.round(2))
coefs.plot.bar(figsize=(10, 5)); plt.title("Coefficients: Linear vs Ridge vs Lasso"); plt.tight_layout(); plt.show()

# %% 6. COMPARE RESULTS
comparison = pd.DataFrame([res_lr,
                           evaluate("Ridge", ridge, X_train_s, X_test_s),
                           evaluate("Lasso", lasso, X_train_s, X_test_s)]).set_index("Model")
print(comparison.round(4))
comparison[["Train R2", "Test R2"]].plot.bar(figsize=(7, 4), ylim=(0.6, 0.9)); plt.title("R2 comparison"); plt.show()
