import pandas as pd
import os

# --- Rutas a tus CSV ---
csv1_path = 'train_2.csv'
csv2_path = 'test2.csv'

# --- Detectar columna con nombres de imagen ---
def extraer_nombres(csv_path):
    df = pd.read_csv(csv_path)
    posibles = [col for col in df.columns if 'img' in col.lower() or 'path' in col.lower()]
    if not posibles:
        raise ValueError(f"No se encontró una columna de imagen en {csv_path}")
    col = posibles[0]
    return df[col].apply(lambda x: os.path.basename(str(x))).tolist()

# --- Cargar nombres ---
nombres1 = set(extraer_nombres(csv1_path))
nombres2 = set(extraer_nombres(csv2_path))

# --- Comparar ---
interseccion = nombres1 & nombres2

print(f"🟢 {len(interseccion)} imágenes .jpg con el mismo nombre en ambos CSVs")
print(f"📁 Ejemplos:")
for nombre in list(interseccion)[:10]:
    print("   ", nombre)

python eval.py  --data_dir Magface_v2/datasets/Casia-Fasd/val_img/color  --model_path checkpoints/casia_v1_best_model.pth  --batch_size 32   --num_classes 2   --ms_layers layer1,layer2
