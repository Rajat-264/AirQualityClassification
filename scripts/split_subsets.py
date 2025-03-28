import pandas as pd
from sklearn.model_selection import train_test_split

datasets = {
    "D1": "C:/AirQualityClassification/data/AirQualityUCI-D1.csv",
    "D2": "C:/AirQualityClassification/data/AirQualityUCI-D2.csv",
    "D3": "C:/AirQualityClassification/data/AirQualityUCI-D3.csv"
}

for key, path in datasets.items():
    df = pd.read_csv(path)

    X = df.drop(columns=["AirQualityCategory"])
    y = df["AirQualityCategory"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(f"C:/AirQualityClassification/data/{key}-train.csv", index=False)
    test_df.to_csv(f"C:/AirQualityClassification/data/{key}-test.csv", index=False)

    print(f"✅ {key}-train.csv and {key}-test.csv created successfully!")
