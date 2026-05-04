import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseVal"] = housing.target
df.to_csv("data/raw/california_housing.csv", index=False)
print(f"Guardado: {len(df)} filas")