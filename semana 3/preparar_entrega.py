import os
import zipfile

# 1. Definir los nombres de los archivos
archivos_a_incluir = [
    'entrenar_moda.py', 
    'fashion_mnist_optimizado.keras', 
    'camara_ropa.py'
]

nombre_zip = 'Entrega_IA_TiempoReal.zip'

def crear_zip():
    print("--- Iniciando empaquetado ---")
    archivos_validos = []
    for f in archivos_a_incluir:
        if os.path.exists(f):
            archivos_validos.append(f)
        else:
            print(f"⚠️ Advertencia: No se encontró '{f}', se saltará.")

    if not archivos_validos:
        print("❌ Error: No hay archivos para comprimir.")
        return

    with zipfile.ZipFile(nombre_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for archivo in archivos_validos:
            zipf.write(archivo)
            print(f"✅ Agregado: {archivo}")

    print(f"\n¡ÉXITO! Se ha generado el archivo: {os.path.abspath(nombre_zip)}")

if __name__ == "__main__":
    crear_zip()