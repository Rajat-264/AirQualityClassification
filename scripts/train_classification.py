import pandas as pd

df = pd.read_csv("C:\AirQualityClassification\data\AirQualityUCI-train.csv", encoding="ISO-8859-1")

def classify_air_quality(value):
    if value < 7.8:
        return "Good"
    elif 7.8 <= value < 16.3:
        return "Moderate"
    else:
        return "Poor"

df["AirQualityCategory"] = df["C6H6(GT)"].apply(classify_air_quality)

df.to_csv("C:\AirQualityClassification\data\AirQualityUCI-train-classified.csv", index=False)
print("✅ New classification labels added successfully!")
