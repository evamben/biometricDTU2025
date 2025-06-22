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
import os

# --- Leer archivo feat.list y cargar magnitudes ---
with open('preprocessing_v1/Casia_Fasd/feat.list', 'r') as f:
    lines = f.readlines()

img_2_mag = {}
for line in lines:
    parts = line.strip().split(' ')
    imgname = parts[0].replace('\\', '/')
    feats = np.array([float(e) for e in parts[1:]], dtype=np.float32)
    mag = np.linalg.norm(feats)
    img_2_mag[imgname] = mag

imgnames = list(img_2_mag.keys())
mags = np.array([img_2_mag[name] for name in imgnames])

# --- Calcular percentiles 33 y 66 ---
p33 = np.percentile(mags, 33)
p66 = np.percentile(mags, 66)

print(f"Percentiles: 33={p33:.3f}, 66={p66:.3f}")

# --- Definir 3 rangos de calidad ---
quality_ranges = {
    'low_quality': mags < p33,
    'medium_quality': (mags >= p33) & (mags < p66),
    'high_quality': mags >= p66
}

# --- Separar imágenes por calidad ---
quality_splits = {}
for quality_name, mask in quality_ranges.items():
    quality_splits[quality_name] = [imgnames[i] for i in np.where(mask)[0]]
    print(f"{quality_name}: {len(quality_splits[quality_name])} imágenes")

# --- Guardar listas por calidad ---
output_dir = 'quality_splits'
os.makedirs(output_dir, exist_ok=True)

for quality_name, img_list in quality_splits.items():
    filepath = os.path.join(output_dir, f"{quality_name}.txt")
    with open(filepath, 'w') as f:
        for imgname in img_list:
            f.write(f"{imgname}\n")
    print(f"Guardado {len(img_list)} imágenes en {filepath}")

