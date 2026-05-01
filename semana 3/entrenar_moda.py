import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Cargar datos
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# 2. AGRUPAR HACIA PRENDAS ESPECÍFICAS
def agrupar_a_prendas_unicas(labels):
    # Mapa: Original -> Nueva Prenda Representativa
    # 0: Camiseta, 1: Pantalón, 2: Vestido, 3: Botín, 4: Bolso
    mapa = {0:0, 2:0, 4:0, 6:0,  # Camiseta, Pullover, Abrigo, Camisa -> TODO ES 'CAMISETA'
            1:1,                 # Pantalón -> PANTALÓN
            3:2,                 # Vestido -> VESTIDO
            5:3, 7:3, 9:3,       # Sandalia, Zapatilla, Botín -> TODO ES 'BOTÍN'
            8:4}                 # Bolso -> BOLSO
    return np.array([mapa[l] for l in labels])

train_labels = agrupar_a_prendas_unicas(train_labels)
test_labels = agrupar_a_prendas_unicas(test_labels)

# Preparar imágenes
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255.0

# 3. Modelo Robusto
model = keras.models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.RandomContrast(0.2), # Ayuda con el brillo del celular
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax') 
])

model.compile(optimizer=keras.optimizers.Adam(0.0005), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("🚀 Entrenando con 5 prendas maestras...")
model.fit(train_images, train_labels, epochs=15, batch_size=64, validation_split=0.2)

model.save('fashion_mnist_optimizado.keras')
print("✅ IA entrenada con éxito.")