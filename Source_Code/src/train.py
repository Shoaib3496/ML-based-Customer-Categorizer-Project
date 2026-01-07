import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/processed/processed_data.csv", index_col=0)
X = df.drop("Close", axis=1)
y = df["Close"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

joblib.dump(model, "models/final_model.pkl")
