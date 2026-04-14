import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
dataset = pd.read_csv("Artificial_Neural_Network_Case_Study_data.csv")

# Display basic info
print(dataset.head())
print(f"Dataset shape: {dataset.shape}")

# Splitting features & target
features = dataset.iloc[:, 3:-1]
target = dataset.iloc[:, -1]

# Encoding Gender column
le = LabelEncoder()
features.iloc[:, 2] = le.fit_transform(features.iloc[:, 2])

# One-hot encoding Geography
transformer = ColumnTransformer(
    transformers=[("geo", OneHotEncoder(drop="first"), [1])],
    remainder="passthrough"
)

features = transformer.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, random_state=0
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build ANN model
ann_model = Sequential([
    Dense(8, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile model
ann_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
ann_model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

# Predictions
pred_probs = ann_model.predict(X_test)
predictions = (pred_probs > 0.5).astype(int)

# Evaluation
print("\n--- Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
print("Report:\n", classification_report(y_test, predictions))