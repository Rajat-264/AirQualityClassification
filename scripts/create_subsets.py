import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "C:/AirQualityClassification/data/AirQualityUCI-classified.csv"

df = pd.read_csv(file_path)

D1 = df.sample(frac=0.25, random_state=42)
D2 = df.sample(frac=0.50, random_state=42)
D3 = df.sample(frac=0.75, random_state=42)

D1.to_csv("C:/AirQualityClassification/data/AirQualityUCI-D1.csv", index=False)
D2.to_csv("C:/AirQualityClassification/data/AirQualityUCI-D2.csv", index=False)
D3.to_csv("C:/AirQualityClassification/data/AirQualityUCI-D3.csv", index=False)

print("✅ D1, D2, and D3 datasets created successfully!")
