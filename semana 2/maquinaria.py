import cv2
import numpy as np
import datetime
import time
import os
import threading  # <--- CLAVE PARA LA FLUIDEZ
from tensorflow.keras.models import load_model
from supabase import create_client, Client

# --- 0. OPTIMIZACIÓN ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. CONFIGURACIÓN ---
url: str = "https://afftlofezngahioexfhy.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFmZnRsb2Zlem5nYWhpb2V4Zmh5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY5NzQ4NTEsImV4cCI6MjA5MjU1MDg1MX0.dtsgQkzqKLBfhfmIHbtQG1f0_chw_yVjnXNmcVaZQ40"
supabase: Client = create_client(url, key)

model = load_model("keras_model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# --- FUNCIÓN DE SUBIDA EN SEGUNDO PLANO ---
def subir_a_nube(img_para_subir, nombre_archivo, clase, score):
    try:
        # Guardar imagen temporal para este hilo
        temp_path = f"upload_{nombre_archivo}"
        cv2.imwrite(temp_path, img_para_subir)
        
        with open(temp_path, 'rb') as f:
            supabase.storage.from_("evidencias").upload(
                path=nombre_archivo, 
                file=f,
                file_options={"content-type": "image/jpeg"}
            )
        
        url_publica = f"{url}/storage/v1/object/public/evidencias/{nombre_archivo}"
        supabase.table("detecciones").insert({
            "conducta": clase,
            "confianza": score,
            "imagen_url": url_publica
        }).execute()
        
        print(f"\n✅ REGISTRO EXITOSO: {clase}")
        os.remove(temp_path) # Limpiar archivo temporal
    except Exception as e:
        print(f"\n⚠️ Error en segundo plano: {e}")

# --- INICIO ---
camera = cv2.VideoCapture(0)
contador_frames = 0 
class_name = "Escaneando..."
confidence_score = 0.0
ultima_subida = 0 # Para evitar saturar con muchas fotos

print(">>> SISTEMA FLUIDO MUNDO-YACUS INICIADO <<<")

while True:
    ret, image = camera.read()
    if not ret: break

    contador_frames += 1
    display_img = image.copy()

    # Analizar 1 de cada 10 cuadros (IA en segundo plano)
    if contador_frames % 10 == 0:
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        image_normalized = (image_array / 127.5) - 1

        prediction = model.predict(image_normalized, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = float(prediction[0][index])

        # Si es falla (ajustar el index != 0 según tu labels.txt) y han pasado 5s desde la última
        if confidence_score > 0.85 and index != 0 and (time.time() - ultima_subida > 5):
            print(f"!!! DETECTADO: {class_name}. Subiendo...")
            
            now = datetime.datetime.now()
            nombre_nube = f"falla_{now.strftime('%H%M%S_%f')}.jpg"
            
            # Lanzar el proceso de nube en un hilo separado
            # El video NO se detendrá mientras esto ocurre
            hilo = threading.Thread(target=subir_a_nube, args=(display_img.copy(), nombre_nube, class_name, confidence_score))
            hilo.start()
            
            ultima_subida = time.time()

    # UI Fluida
    cv2.rectangle(display_img, (0,0), (320, 80), (0,0,0), -1)
    cv2.putText(display_img, f"STATUS: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display_img, f"CONF: {confidence_score:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("MONITOR INDUSTRIAL - MUNDO YACUS", display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera.release()
cv2.destroyAllWindows()