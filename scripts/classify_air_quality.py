import pandas as pd

file_path = "C:/AirQualityClassification/data/AirQualityUCI-cleaned.csv"
output_path = "C:/AirQualityClassification/data/AirQualityUCI-classified.csv"

df = pd.read_csv(file_path)

def classify_air_quality(value):
    if value < 7.8:
        return "Good"
    elif 7.8 <= value < 16.3:
        return "Moderate"
    else:
        return "Poor"

df["AirQualityCategory"] = df["C6H6(GT)"].apply(classify_air_quality)

df.to_csv(output_path, index=False)
print(f"✅ AirQualityCategory column added. Classified dataset saved at: {output_path}")
