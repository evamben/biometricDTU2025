# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import os

# # Leer líneas del archivo feat.list
# with open('preprocessing_v1/Casia_Fasd/feat.list', 'r') as f:
#     lines = f.readlines()

# img_2_feats = {}
# img_2_mag = {}

# # Procesar cada línea para extraer nombre de imagen y vector de características
# for line in lines:
#     parts = line.strip().split(' ')
#     imgname = parts[0].replace('\\', '/')  # Normalizar separadores
#     feats = np.array([float(e) for e in parts[1:]], dtype=np.float32)
#     mag = np.linalg.norm(feats)
#     img_2_feats[imgname] = feats / mag  # Normalizar vector
#     img_2_mag[imgname] = mag

# imgnames = list(img_2_mag.keys())
# mags = np.array([img_2_mag[imgname] for imgname in imgnames])

# # Definir categorías por magnitud
# small_mask = mags < 25
# medium_mask = (mags >= 25) & (mags <= 33)
# large_mask = mags > 33

# def select_representatives(mask, n=2):
#     indices = np.where(mask)[0]
#     if len(indices) == 0:
#         return []
#     group_mags = mags[indices]
#     target = np.mean(group_mags)
#     diffs = np.abs(group_mags - target)
#     closest_indices = diffs.argsort()[:n]
#     return indices[closest_indices]

# indices_small = select_representatives(small_mask, 2)
# indices_medium = select_representatives(medium_mask, 2)
# indices_large = select_representatives(large_mask, 2)

# indices_seleccionados = np.concatenate([indices_small, indices_medium, indices_large])

# print("Índices seleccionados:", indices_seleccionados)
# print("Magnitudes seleccionadas:", mags[indices_seleccionados])

# # Parámetros para visualización
# H, W = 112, 112
# NH, NW = 2, 3  # 2 filas, 3 columnas

# canvas = np.zeros((NH * H, NW * W, 3), np.uint8)

# for i, idx in enumerate(indices_seleccionados):
#     img_path = imgnames[idx].replace('\\', '/')
#     img_path = os.path.normpath(img_path)  # Normaliza separadores sin agregar base_dir

#     print(f"Intentando abrir imagen: {img_path}")
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"Error leyendo imagen: {img_path}")
#         continue
#     img = cv2.resize(img, (W, H))

#     fila = i // NW
#     columna = i % NW
#     canvas[fila*H:(fila+1)*H, columna*W:(columna+1)*W, :] = img

# plt.figure(figsize=(18, 8))
# plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title("2 imágenes representativas por categoría de magnitud (<25, 25-30, >30)")
# plt.show()
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pyiqa
import torch
from torchvision import transforms

# Crear modelo BRISQUE (más rápido y preciso que NIQE en general)
model_quality = pyiqa.create_metric('brisque', device='cpu')

# Leer líneas del archivo feat.list
with open('preprocessing_v1/Casia_Fasd/feat.list', 'r') as f:
    lines = f.readlines()

img_2_feats = {}
img_2_mag = {}
img_2_quality = {}

# Procesar cada línea para extraer nombre de imagen y vector de características
for line in lines:
    parts = line.strip().split(' ')
    imgname = parts[0].replace('\\', '/')  # Normalizar separadores
    feats = np.array([float(e) for e in parts[1:]], dtype=np.float32)
    mag = np.linalg.norm(feats)
    img_2_feats[imgname] = feats / mag  # Normalizar vector
    img_2_mag[imgname] = mag

imgnames = list(img_2_mag.keys())

# Función para calcular calidad usando pyiqa
def calc_quality_score(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error leyendo imagen: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_t = transform(img).unsqueeze(0)  # batch dim

    with torch.no_grad():
        score = model_quality(img_t)
    return score.item()

print("Calculando scores de calidad para todas las imágenes (esto puede tardar)...")
for imgname in imgnames:
    img_path = os.path.normpath(imgname)
    score = calc_quality_score(img_path)
    if score is None:
        score = np.nan
    img_2_quality[imgname] = score

qualities = np.array([img_2_quality[imgname] for imgname in imgnames])

# Calcular cuartiles basados en scores de calidad
quartiles = np.percentile(qualities[~np.isnan(qualities)], [25, 50, 75])

# Crear máscaras por cuartil de calidad
q1_mask = qualities <= quartiles[0]          # <= Q1 (bajo)
q2_mask = (qualities > quartiles[0]) & (qualities <= quartiles[1])  # Q1-Q2
q3_mask = (qualities > quartiles[1]) & (qualities <= quartiles[2])  # Q2-Q3
q4_mask = qualities > quartiles[2]           # > Q3 (alto)

def select_representatives(mask, n=2):
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []
    group_qualities = qualities[indices]
    target = np.mean(group_qualities)
    diffs = np.abs(group_qualities - target)
    closest_indices = diffs.argsort()[:n]
    return indices[closest_indices]

indices_q1 = select_representatives(q1_mask, 2)
indices_q2 = select_representatives(q2_mask, 2)
indices_q3 = select_representatives(q3_mask, 2)
indices_q4 = select_representatives(q4_mask, 2)

indices_seleccionados = np.concatenate([indices_q1, indices_q2, indices_q3, indices_q4])

print("Índices seleccionados:", indices_seleccionados)
print("Quality scores seleccionados:", qualities[indices_seleccionados])

# Parámetros para visualización
H, W = 112, 112
NH, NW = 2, 4  # 2 filas, 4 columnas

canvas = np.zeros((NH * H, NW * W, 3), np.uint8)

for i, idx in enumerate(indices_seleccionados):
    img_path = os.path.normpath(imgnames[idx])

    print(f"Intentando abrir imagen: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error leyendo imagen: {img_path}")
        continue
    img = cv2.resize(img, (W, H))

    fila = i // NW
    columna = i % NW
    canvas[fila*H:(fila+1)*H, columna*W:(columna+1)*W, :] = img

plt.figure(figsize=(22, 8))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("2 imágenes representativas por cuartiles de calidad (BRISQUE score)")
plt.show()

# Mostrar scores de calidad para imágenes seleccionadas
print("Quality scores para imágenes seleccionadas:")
for idx in indices_seleccionados:
    imgname = imgnames[idx]
    print(f"{imgname}: Quality Score = {img_2_quality[imgname]:.3f}, Magnitud = {img_2_mag[imgname]:.3f}")
