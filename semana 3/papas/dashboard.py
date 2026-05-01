import streamlit as st
import pandas as pd
from supabase import create_client, Client
import plotly.express as px

# --- CONFIGURACIÓN SUPABASE ---
SUPABASE_URL = "https://ltgjmstozkgvbfvgrkhp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx0Z2ptc3RvemtndmJmdmdya2hwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc1ODAyNzYsImV4cCI6MjA5MzE1NjI3Nn0.SrEKHPyJe5Zp0cC4lcotunv3CbFXGwATV2tkByOQMW4"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configuración de página
st.set_page_config(page_title="AgroIA - Control de Cultivos", layout="wide", page_icon="🥔")

# --- ESTILO CSS PERSONALIZADO (Corregido para modo oscuro y visibilidad) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    /* Forzamos colores legibles en las métricas sin importar el tema del navegador */
    [data-testid="stMetricValue"] { color: #1f1f1f !important; }
    [data-testid="stMetricLabel"] { color: #5f6368 !important; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #eeeeee;
    }
    </style>
    """, unsafe_allow_html=True)

# --- TÍTULO Y SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2153/2153067.png", width=100)
    st.title("Panel de Control")
    if st.button('🔄 Actualizar Dashboard'):
        st.rerun()
    st.info("Sistema de monitoreo automático mediante Visión Artificial.")

st.title("🥔 AgroIA: Monitor Fitopatológico de Papa")
st.write("Análisis de salud del cultivo en tiempo real")
st.divider()

# --- LÓGICA DE CARGA DE DATOS ---
try:
    response = supabase.table("historial_papas").select("*").order("fecha", desc=True).execute()
    
    if response.data and len(response.data) > 0:
        df = pd.DataFrame(response.data)
        df['fecha'] = pd.to_datetime(df['fecha'])

        # --- MÉTRICAS ---
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Muestras", len(df))
        with m2:
            sanas = len(df[df['estado'] == 'Sana'])
            st.metric("Hojas Sanas", sanas)
        with m3:
            enfermas = len(df[df['estado'] != 'Sana'])
            st.metric("Alertas Detectadas", enfermas, delta=f"{enfermas} críticas", delta_color="inverse")
        with m4:
            promedio_conf = f"{df['confianza'].mean():.1f}%"
            st.metric("Confianza Media", promedio_conf)

        st.divider()

        # --- GRÁFICOS ---
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("📊 Análisis de Salud")
            fig_pie = px.pie(df, names='estado', hole=0.4, 
                            color='estado',
                            color_discrete_map={'Sana':'#2ecc71', 'Tardia':'#e74c3c', 'Temprana':'#f1c40f', 'Desecho':'#95a5a6'})
            st.plotly_chart(fig_pie, use_container_width=True)

        with g2:
            st.subheader("📈 Evolución de la Detección")
            st.area_chart(df.set_index('fecha')['confianza'])

        # --- TABLA (Corregida con width='stretch') ---
        st.subheader("📋 Registro de Actividad Reciente")
        st.dataframe(df[['fecha', 'estado', 'confianza']].head(15), width='stretch', hide_index=True)

    else:
        st.warning("⚠️ Conexión establecida, pero no se encontraron registros en 'historial_papas'.")
        st.info("💡 Por favor, inicia tu script de cámara y asegúrate de que la IA detecte algo.")

except Exception as e:
    st.error(f"❌ Error al conectar con Supabase: {e}")