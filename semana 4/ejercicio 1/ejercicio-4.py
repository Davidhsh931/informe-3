import streamlit as st
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
from supabase import create_client, Client

# --- 1. Configuración de la Página Streamlit ---
st.set_page_config(page_title="Monitor de Tráfico IA", layout="wide")
st.title("🛡️ Sistema de Control de Tráfico en Tiempo Real")

st.sidebar.header("📊 Estadísticas de Sesión")
stats_placeholder = st.sidebar.empty()
unique_count_text = st.sidebar.empty()
db_status_placeholder = st.sidebar.empty() # Para mostrar cuando envía a DB

# --- 2. Inicialización de Supabase ---
SUPABASE_URL = 'https://ypktexzpugbqvglqirbb.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlwa3RleHpwdWdicXZnbHFpcmJiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgxNzg1NzUsImV4cCI6MjA5Mzc1NDU3NX0.LyHbAY-juJnladWY97H2BV8xSSkd0g-9aPCOToWbLQ8' 

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- 3. Carga del Modelo YOLO ---
@st.cache_resource
def load_model():
    # Usamos nano para máxima velocidad en tiempo real
    return YOLO('yolov8n.pt')

model = load_model()

# --- 4. Configuración de Zona y Clases ---
# Coordenadas de la caja azul [x_min, y_min, x_max, y_max]
FORBIDDEN_ZONE = [50, 50, 450, 450]
# Clases COCO: 2(car), 3(motorcycle), 5(bus), 7(truck)
VEHICLE_CLASSES = [2, 3, 5, 7] 

def is_inside_forbidden_zone(bbox, zone):
    x_min_obj, y_min_obj, x_max_obj, y_max_obj = bbox
    x_min_zone, y_min_zone, x_max_zone, y_max_zone = zone
    # Lógica de intersección: Retorna True si el objeto toca la zona
    return not (x_max_obj < x_min_zone or x_min_obj > x_max_zone or
                y_max_obj < y_min_zone or y_min_obj > y_max_zone)

def upload_to_supabase(unique_total, current_in_zone):
    try:
        data = {
            'date': datetime.now().date().isoformat(),
            'unique_vehicles': unique_total,    # Acumulado del día
            'total_vehicles': current_in_zone,  # Autos en la zona en este instante
            'zone_coordinates': [str(val) for val in FORBIDDEN_ZONE]
        }
        supabase.table('traffic_counts').insert(data).execute()
        return True
    except Exception as e:
        print(f"Error Supabase: {e}")
        return False

# --- 5. Lógica Principal ---
def main():
    col1, col2 = st.columns([3, 1])
    with col1:
        frame_placeholder = st.empty()
    with col2:
        st.info("💡 **Reglas de detección:**\n\nSolo se cuentan vehículos (Autos, Motos, Buses, Camiones) que ingresen dentro del recuadro azul.")
        run = st.checkbox('🔴 Iniciar Monitoreo', value=True)
    
    cap = cv2.VideoCapture(0)
    
    ids_contados = set() # Set para asegurar que no haya IDs duplicados
    last_upload_time = 0
    upload_cooldown = 10 # Enviar a Supabase cada 10 segundos

    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            st.error("Error al leer la cámara.")
            break

        # YOLO Tracking (persist=True es clave para mantener los IDs)
        results = model.track(frame, persist=True, conf=0.5, verbose=False)[0]
        current_frame_count = 0

        # Dibujar Zona de Control (Azul)
        cv2.rectangle(frame, (FORBIDDEN_ZONE[0], FORBIDDEN_ZONE[1]), 
                      (FORBIDDEN_ZONE[2], FORBIDDEN_ZONE[3]), (255, 0, 0), 3)

        # Si detecta objetos con ID
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            ids = results.boxes.id.cpu().numpy().astype(int)
            clases = results.boxes.cls.cpu().numpy().astype(int)

            for box, obj_id, cls in zip(boxes, ids, clases):
                # Filtrar solo si es un vehículo
                if cls in VEHICLE_CLASSES:
                    # Verificar si entra en la zona azul
                    if is_inside_forbidden_zone(box, FORBIDDEN_ZONE):
                        current_frame_count += 1
                        ids_contados.add(obj_id) # Se añade al set (ignora si ya existe)
                        
                        # Dibujar caja amarilla sobre el auto y su ID
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
                        cv2.putText(frame, f"ID:{obj_id}", (box[0], box[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Subida a Supabase
        curr_time = time.time()
        if curr_time - last_upload_time > upload_cooldown:
            if len(ids_contados) > 0:
                if upload_to_supabase(len(ids_contados), current_frame_count):
                    db_status_placeholder.success(f"✅ Sincronizado: {datetime.now().strftime('%H:%M:%S')}")
                last_upload_time = curr_time

        # Renderizar Imagen en Streamlit (Usando width='stretch' para 2026+)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')
        
        # Actualizar Métricas Laterales
        stats_placeholder.metric("🚙 En zona ahora", current_frame_count)
        unique_count_text.metric("📈 Vehículos Únicos (Día)", len(ids_contados))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()