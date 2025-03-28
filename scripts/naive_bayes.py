import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_df = pd.read_csv("C:\AirQualityClassification\data\AirQualityUCI-train-classified.csv")
test_df = pd.read_csv("C:\AirQualityClassification\data\AirQualityUCI-test-classified.csv")

X_train = train_df.drop(columns=["AirQualityCategory", "C6H6(GT)"])
y_train = train_df["AirQualityCategory"]
X_test = test_df.drop(columns=["AirQualityCategory", "C6H6(GT)"])
y_test = test_df["AirQualityCategory"]

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("✅ Naive Bayes Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))