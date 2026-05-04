import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("california-housing-prediction")

# Cargar datos
df = pd.read_csv("data/raw/california_housing.csv")
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos a probar
experiments = [
    (Ridge(alpha=0.01), "Ridge_0.01", {"alpha": 0.01}),
    (Ridge(alpha=0.1), "Ridge_0.1", {"alpha": 0.1}),
    (Ridge(alpha=1.0), "Ridge_1.0", {"alpha": 1.0}),
    (RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42), "RF_50_5", {"n_estimators": 50, "max_depth": 5}),
    (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), "RF_100_10", {"n_estimators": 100, "max_depth": 10}),
    (RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42), "RF_150_15", {"n_estimators": 150, "max_depth": 15}),
    (GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42), "GB_50_3", {"n_estimators": 50, "max_depth": 3}),
    (GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42), "GB_100_5", {"n_estimators": 100, "max_depth": 5}),
    (GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42), "GB_100_5_lr05", {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05}),
]

best_model, best_scaler, best_r2, best_name = None, None, -999, ""

for model, name, params in experiments:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    with mlflow.start_run(run_name=name):
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")
    
    print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    if r2 > best_r2:
        best_r2, best_model, best_scaler, best_name = r2, model, scaler, name

os.makedirs("models", exist_ok=True)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump({"model": best_model, "scaler": best_scaler}, f)

print(f"\n🏆 Mejor: {best_name} (R²={best_r2:.4f})")
print("💾 Guardado en models/best_model.pkl")