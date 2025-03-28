import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("C:/AirQualityClassification/data/AirQualityUCI-train-classified.csv")

X = df.drop(columns=["AirQualityCategory"])
y = df["AirQualityCategory"].map({"Good": 0, "Moderate": 1, "Poor": 2})  # Encode categories

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def train_ann(activation_function):
    model = keras.Sequential([
        keras.layers.Dense(16, activation=activation_function, input_shape=(X_train.shape[1],)),
        keras.layers.Dense(8, activation=activation_function),
        keras.layers.Dense(3, activation="softmax")  
    ])
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)
    return history

activations = ["relu", "sigmoid", "tanh"]
history_dict = {}
for activation in activations:
    print(f"Training with {activation} activation...")
    history_dict[activation] = train_ann(activation)

plt.figure(figsize=(10, 6))
for activation in activations:
    plt.plot(history_dict[activation].history["loss"], label=f"{activation} - Train")
    plt.plot(history_dict[activation].history["val_loss"], linestyle="dashed", label=f"{activation} - Validation")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Function Values Against Epochs")
plt.legend()
plt.show()

print("✅ ANN models trained successfully. Loss curves plotted!")
