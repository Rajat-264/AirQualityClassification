import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

datasets = ["D1", "D2", "D3", "AirQualityUCI"]

for dataset in datasets:
    train_path = f"C:/AirQualityClassification/data/{dataset}-train.csv"
    test_path = f"C:/AirQualityClassification/data/{dataset}-test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop(columns=["AirQualityCategory"])
    y_train = train_data["AirQualityCategory"]
    X_test = test_data.drop(columns=["AirQualityCategory"])
    y_test = test_data["AirQualityCategory"]

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"\n📊 Results for {dataset}:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
