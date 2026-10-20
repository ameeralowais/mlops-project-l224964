import os
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

STUDENT_ID = "l224964"
DATA_PATH = "data/dataset.csv"
N_ESTIMATORS = 100 
MAX_DEPTH = 10

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
else:
    print(f"[{STUDENT_ID}] data/dataset.csv not found, using sklearn housing data")
    X, y = fetch_california_housing(return_X_y=True)

print(f"[{STUDENT_ID}] Loaded data with shape {X.shape}")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42).fit(X_tr, y_tr)
print(f"[{STUDENT_ID}] R2 score: {model.score(X_te, y_te):.3f}")

os.makedirs("model", exist_ok=True)
joblib.dump(model, f"model/model_{STUDENT_ID}.pkl")