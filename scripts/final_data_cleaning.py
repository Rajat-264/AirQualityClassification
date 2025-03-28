import pandas as pd

file_path = "C:\AirQualityClassification\data\AirQualityUCI-intermediate.csv"  
output_path = "C:\AirQualityClassification\data\AirQualityUCI-cleaned.csv"

def clean_air_quality_data(file_path, output_path):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    
    columns_to_keep = ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)","C6H6(GT)", 
                       "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", 
                       "T", "RH", "AH"]
    
    df = df[columns_to_keep]

    df = df.apply(pd.to_numeric, errors="coerce")
    
    df = df.dropna(axis=1, thresh=len(df) * 0.5)

    df = df.dropna(axis=0, thresh=df.shape[1] - 2)

    df = df.apply(lambda col: col.fillna((col.shift(1) + col.shift(-1)) / 2) if col.dtype != "object" else col)

    df.fillna(df.mean(), inplace=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Processed dataset saved successfully at: {output_path}")

clean_air_quality_data(file_path, output_path)
