import pandas as pd
import numpy as np
import os

# --- Configuración: pon tus rutas aquí ---
feat_list_path = 'feat.list'  # o .npy si es NumPy
csv_path = 'test_2.csv'

# --- Cargar lista de imágenes del feat_list ---
with open(feat_list_path, 'r', encoding='utf-8') as f:
    feat_images = [os.path.basename(line.strip().split()[0]) for line in f if line.strip()]


print(f"🔍 {len(feat_images)} imágenes en feat_list")

# --- Cargar CSV ---
df = pd.read_csv(csv_path)

# Detectar la columna con las rutas o nombres de imagen (asumimos que se llama 'image' o similar)
column_candidates = [col for col in df.columns if 'img' in col.lower() or 'path' in col.lower()]
if not column_candidates:
    raise ValueError("❌ No se encontró una columna de imagen en el CSV")
image_column = column_candidates[0]

# Extraer y normalizar nombres de imagen desde CSV
csv_images = df[image_column].apply(lambda x: os.path.basename(str(x))).tolist()
csv_images_set = set(csv_images)

# --- Comparación ---
missing = [img for img in feat_images if img not in csv_images_set]
present = [img for img in feat_images if img in csv_images_set]

print(f"✅ {len(present)} imágenes están en el CSV")
print(f"❌ {len(missing)} imágenes NO están en el CSV")

# (Opcional) Mostrar las que faltan
if missing:
    print("\nEjemplos de imágenes que faltan:")
    print(len(missing))

