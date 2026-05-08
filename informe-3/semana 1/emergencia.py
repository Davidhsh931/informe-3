import os
# Silenciar advertencias de TensorFlow antes de importar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from supabase import create_client

# --- CONFIGURACIÓN SUPABASE ---
URL = "https://yrsdkwbijbhajrdaazmr.supabase.co"
KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlyc2Rrd2JpamJoYWpyZGFhem1yIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzYzNzQ4MDksImV4cCI6MjA5MTk1MDgwOX0.iOxK8HQgWZ1o5mLVks166_KzQtg61tU6dA1wK8jfNGc"
supabase = create_client(URL, KEY)

# Cargar Modelo TFLite (Usando API de bajo nivel para evitar conflictos)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_unquant.tflite")
LABEL_PATH = os.path.join(BASE_DIR, "labels.txt")

with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Variables para Regresión
historial_minutos = {} 
ultima_subida = 0

def calcular_regresion(datos):
    if len(datos) < 2: return 0, 0
    X = np.array(list(datos.keys())).astype(float)
    Y = np.array(list(datos.values())).astype(float)
    n = len(X)
    # Fórmulas de Mínimos Cuadrados
    b = (n * np.sum(X*Y) - np.sum(X) * np.sum(Y)) / (n * np.sum(X**2) - (np.sum(X))**2)
    a = (np.sum(Y) - b * np.sum(X)) / n
    return a, b

def iniciar_sistema():
    global ultima_subida
    
    # Carga segura del intérprete
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(0)
    print("🚀 Sistema de Vigilancia Mundo-Yacus Activo...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Preprocesamiento de Imagen
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = ImageOps.fit(Image.fromarray(img_rgb), (224, 224), Image.Resampling.LANCZOS)
        img_arr = (np.asarray(img_pil).astype(np.float32) / 127.5) - 1
        
        # Inferencia de IA
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img_arr, axis=0))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        idx = np.argmax(output[0])
        fase = labels[idx]
        conf = float(output[0][idx] * 100)

        t_actual = time.time()
        minuto_actual = int((t_actual % 3600) // 60) # Minuto relativo para la gráfica

        # Registro de Incidencia (Mínimo 85% de confianza)
        if conf > 85.0 and "Fondo" not in fase:
            if t_actual - ultima_subida > 4: # Evitar spam de subidas
                img_name = f"incidencia_{int(t_actual)}.jpg"
                cv2.imwrite(img_name, frame)
                
                # Subir a la Nube
                try:
                    with open(img_name, 'rb') as f:
                        supabase.storage.from_('fotos-incidencias').upload(img_name, f)
                    img_url = supabase.storage.from_('fotos-incidencias').get_public_url(img_name)

                    supabase.table("incidencias_transito").insert({
                        "conducta": fase, "confianza": conf, "imagen_url": img_url
                    }).execute()
                    
                    historial_minutos[minuto_actual] = historial_minutos.get(minuto_actual, 0) + 1
                    ultima_subida = t_actual
                    
                    a, b = calcular_regresion(historial_minutos)
                    print(f"🔔 REGISTRADO: {fase} | Tendencia (Regresión): Y = {a:.2f} + {b:.2f}X")
                except Exception as e:
                    print(f"Error de conexión: {e}")

        # Interfaz Visual
        cv2.rectangle(frame, (0,0), (450, 80), (0,0,0), -1)
        cv2.putText(frame, f"Conducta: {fase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Confianza: {conf:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        cv2.imshow("MUNDO-YACUS: DASHBOARD TRANSITO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_sistema()