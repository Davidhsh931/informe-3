import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Configuración
IMG_SIZE = 128 # Subimos resolución para ver mejor las manchas
BATCH_SIZE = 32
DATA_PATH = 'dataset_examen' # La carpeta que creaste

# 1. Cargar imágenes
train_ds = keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

clases = train_ds.class_names
print(f"✅ Entrenando para detectar: {clases}")

# 2. Modelo Profesional
model = keras.models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Rescaling(1./255),
    
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(clases), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Entrenar
print("🚀 Iniciando entrenamiento de fitopatología...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# 4. Guardar
model.save('modelo_papas_examen.keras')
print("✅ ¡Listo! Modelo 'modelo_papas_examen.keras' generado.")