import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from supabase import create_client, Client

# --- CONFIGURACIÓN SUPABASE ---
SUPABASE_URL = "https://ltgjmstozkgvbfvgrkhp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx0Z2ptc3RvemtndmJmdmdya2hwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc1ODAyNzYsImV4cCI6MjA5MzE1NjI3Nn0.SrEKHPyJe5Zp0cC4lcotunv3CbFXGwATV2tkByOQMW4"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- CONFIGURACIÓN MODELO ---
model = keras.models.load_model('modelo_papas_examen.keras')
nombres_clases = ['Desecho', 'Sana', 'Tardia', 'Temprana'] 

# Variables para control de envío (Anti-spam)
ultima_vez_enviado = 0
intervalo_envio = 5 # Segundos entre cada guardado en DB

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    x1, y1, x2, y2 = w//2-150, h//2-150, w//2+150, h//2+150
    roi = frame[y1:y2, x1:x2]

    # Procesamiento
    img_ia = cv2.resize(roi, (128, 128))
    img_predict = cv2.cvtColor(img_ia, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_predict, axis=0)

    # Predicción
    pred = model.predict(img_array, verbose=0)
    indice = np.argmax(pred[0])
    conf = np.max(pred[0]) * 100
    clase = nombres_clases[indice]

    # --- LÓGICA DE ENVÍO A SUPABASE ---
    tiempo_actual = time.time()
    # Si la IA está muy segura (>90%) y pasaron 5 segundos, guardamos
    if conf > 90 and (tiempo_actual - ultima_vez_enviado) > intervalo_envio:
        try:
            data = {
                "estado": clase,
                "confianza": float(conf)
            }
            supabase.table("historial_papas").insert(data).execute()
            print(f"☁️ Guardado en Supabase: {clase} ({conf:.1f}%)")
            ultima_vez_enviado = tiempo_actual
        except Exception as e:
            print(f"❌ Error al conectar con Supabase: {e}")

    # --- INTERFAZ ---
    color_borde = (0, 255, 0) if clase == 'Sana' else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_borde, 2)
    cv2.putText(frame, f"{clase} {conf:.1f}%", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_borde, 2)

    cv2.imshow('Examen: Detector + Supabase', frame)
    cv2.imshow('Cerebro IA (Monitor)', cv2.resize(img_ia, (300, 300)))

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()