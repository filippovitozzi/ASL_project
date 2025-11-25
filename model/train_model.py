import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# === 1. Caricamento dataset ===
train_df = pd.read_csv(r"C:\Users\filip\OneDrive\Desktop\ASL_project\dataset\sign_mnist_train.csv")
test_df  = pd.read_csv(r"C:\Users\filip\OneDrive\Desktop\ASL_project\dataset\sign_mnist_test.csv")

X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_train = train_df.iloc[:, 0].values

X_test  = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_test  = test_df.iloc[:, 0].values

num_classes = int(np.max(y_train)) + 1
y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
print("Classi:", num_classes)

