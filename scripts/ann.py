import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_df = pd.read_csv("C:\AirQualityClassification\data\AirQualityUCI-train-classified.csv")
test_df = pd.read_csv("C:\AirQualityClassification\data\AirQualityUCI-test-classified.csv")

X_train = train_df.drop(columns=["AirQualityCategory", "C6H6(GT)"])
y_train = train_df["AirQualityCategory"]
X_test = test_df.drop(columns=["AirQualityCategory", "C6H6(GT)"])
y_test = test_df["AirQualityCategory"]

model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=500, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("✅ ANN Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))