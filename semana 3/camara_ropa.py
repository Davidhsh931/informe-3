import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Cargar el modelo
model = keras.models.load_model('fashion_mnist_optimizado.keras')

# NOMBRES DE PRENDAS ESPECÍFICAS
nombres_clases = ['Camiseta', 'Pantalon', 'Vestido', 'Botin', 'Bolso']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1) # Efecto espejo

    # Zona de detección
    height, width, _ = frame.shape
    x1, y1, x2, y2 = width//2-150, height//2-150, width//2+150, height//2+150
    roi = frame[y1:y2, x1:x2]

    # Procesamiento para pantalla de celular
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    resized = cv2.resize(blurred, (28, 28))
    
    # Inversión de colores (importante para fondo blanco de celular)
    inverted = cv2.bitwise_not(resized)
    _, cleaned = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    input_img = (cleaned / 255.0).reshape(1, 28, 28, 1)

    # Predicción
    pred = model.predict(input_img, verbose=0)
    indice = np.argmax(pred[0])
    conf = np.max(pred[0]) * 100

    # Interfaz
    color = (0, 255, 0) if conf > 85 else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y1-35), (x1+220, y1), color, -1)
    cv2.putText(frame, f"{nombres_clases[indice]} {conf:.1f}%", (x1+5, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Detector de Prendas Especializadas', frame)
    cv2.imshow('Cerebro IA (28x28)', cv2.resize(cleaned, (150, 150)))

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()