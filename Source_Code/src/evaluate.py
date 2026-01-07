import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/processed/processed_data.csv", index_col=0)
X = df.drop("Close", axis=1)
y = df["Close"]

model = joblib.load("models/final_model.pkl")
preds = model.predict(X)

print("MAE:", mean_absolute_error(y, preds))
print("RMSE:", mean_squared_error(y, preds, squared=False))
print("R2:", r2_score(y, preds))
