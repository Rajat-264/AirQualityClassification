import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\AirQualityClassification\data\AirQualityUCI.csv"
# Load the dataset
df = pd.read_csv(file_path, encoding="ISO-8859-1")  # Update filename if needed

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

# Show first few rows
print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())


