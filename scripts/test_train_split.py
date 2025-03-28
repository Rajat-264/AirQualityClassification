import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "C:\AirQualityClassification\data\AirQualityUCI-cleaned.csv"  
output_path_train = "C:\AirQualityClassification\data\AirQualityUCI-train.csv"
output_path_test = "C:\AirQualityClassification\data\AirQualityUCI-test.csv"

def clean_air_quality_data(file_path):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    
    columns_to_keep =["CO(GT)","PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", 
                       "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", 
                       "T","RH","AH"]
    
    df = df[columns_to_keep]
    
    df = df.apply(pd.to_numeric, errors="coerce")
    
    df = df.apply(lambda col: col.fillna((col.shift(1) + col.shift(-1)) / 2) if col.dtype != "object" else col)
    
    df.fillna(df.mean(), inplace=True)
    
    return df

def split_data(df, output_path_train, output_path_test):
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    
    train_set.to_csv(output_path_train, index=False)
    test_set.to_csv(output_path_test, index=False)
    
    print(f"Training dataset saved at: {output_path_train}")
    print(f"Testing dataset saved at: {output_path_test}")

df_cleaned = clean_air_quality_data(file_path)
split_data(df_cleaned, output_path_train, output_path_test)
