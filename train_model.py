import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 1. Load Dataset
print("Loading dataset from train.csv.zip...")
df = pd.read_csv('train.csv.zip')

# 2. Preprocess
y = df.iloc[:, 0].values  # Labels
X = df.iloc[:, 1:].values # Pixels

# Reshape to 28x28 and Normalize
X = X.reshape(-1, 28, 28, 1) / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 3. Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Train
print("Training AI... (This will take 1-2 minutes)")
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# 5. Save the file that app.py is looking for
model.save('digit_model.h5')
print("✅ Success! 'digit_model.h5' has been created in your folder.")
