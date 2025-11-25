import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === 1. Caricamento dataset ===
train_df = pd.read_csv(r"C:\Users\filip\OneDrive\Desktop\ASL_project\dataset\sign_mnist_train.csv")
test_df  = pd.read_csv(r"C:\Users\filip\OneDrive\Desktop\ASL_project\dataset\sign_mnist_test.csv")

X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_train = train_df.iloc[:, 0].values

datagen = ImageDataGenerator(
    rotation_range=10,        # non esagerare su 28x28
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)

datagen.fit(X_train)


X_test  = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_test  = test_df.iloc[:, 0].values

num_classes = int(np.max(y_train)) + 1
y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"""
Dataset shapes:
  ▸ Train: {X_train.shape}
  ▸ Val:   {X_val.shape}
  ▸ Test:  {X_test.shape}

Numero classi: {num_classes}
""")

# === 1.2 Data augmentation mirata sulle classi difficili ===

# classi difficili identificate dal classification_report (le tue)
hard_classes = [12, 13, 17, 19, 24]

# etichette integer per il train (da one-hot → argmax)
y_train_int = np.argmax(y_train, axis=1)

# maschera per selezionare solo esempi delle classi difficili
mask_hard = np.isin(y_train_int, hard_classes)
X_hard = X_train[mask_hard]
y_hard = y_train[mask_hard]

print("Campioni nelle classi difficili:", X_hard.shape[0])

# ImageDataGenerator più aggressivo SOLO per le classi difficili
hard_augmenter = ImageDataGenerator(
    rotation_range=10,       # prima era 20
    width_shift_range=0.15,  # prima era 0.25
    height_shift_range=0.15, # prima era 0.25
    zoom_range=0.15,         # prima era 0.25
    shear_range=0.10         # prima era 0.20
)

# genera nuove immagini augmentate per le classi difficili
augmented_images = []
augmented_labels = []

num_aug_per_sample = 1  # quante nuove immagini creare per ogni esempio difficile

for i in range(len(X_hard)):
    img = X_hard[i]
    label = y_hard[i]

    img = img.reshape((1, 28, 28, 1))  # batch di 1 per flow

    aug_iter = hard_augmenter.flow(img, batch_size=1)
    for _ in range(num_aug_per_sample):
        aug_img = next(aug_iter)[0]
        augmented_images.append(aug_img)
        augmented_labels.append(label)

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

print("Nuove immagini augmentate create:", augmented_images.shape[0])

# aggiungiamo i campioni augmentati al training set
X_train = np.concatenate([X_train, augmented_images])
y_train = np.concatenate([y_train, augmented_labels])

print(f"""
Dataset shapes (dopo augmentation mirata):
  ▸ Train: {X_train.shape}
  ▸ Val:   {X_val.shape}
  ▸ Test:  {X_test.shape}
""")

# === 2. Definizione del modello CNN baseline ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.7),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === 3. Addestramento modello ===

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=3,              
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_signmnist_cnn.h5",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint]
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f}  -  Test accuracy: {test_acc:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
print(cm)

model.save("asl_signmnist_final.keras")
