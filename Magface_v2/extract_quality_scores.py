import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Leer líneas del archivo feat.list
with open('toy_imgs/feat.list', 'r') as f:
    lines = f.readlines()

img_2_feats = {}
img_2_mag = {}

# Procesar cada línea para extraer nombre de imagen y vector de características
for line in lines:
    parts = line.strip().split(' ')
    imgname = parts[0]
    feats = np.array([float(e) for e in parts[1:]], dtype=np.float32)
    mag = np.linalg.norm(feats)
    img_2_feats[imgname] = feats / mag  # Normalizar vector
    img_2_mag[imgname] = mag

imgnames = list(img_2_mag.keys())
mags = [img_2_mag[imgname] for imgname in imgnames]

# Ordenar índices de imágenes por la magnitud (de menor a mayor)
sort_idx = np.argsort(mags)

# Parámetros para visualización
H, W = 112, 112  # tamaño de cada imagen en el canvas (ajustar según tus imágenes)
NH, NW = 1, 10   # filas y columnas del canvas

canvas = np.zeros((NH * H, NW * W, 3), np.uint8)

# Construir el canvas con imágenes ordenadas por magnitud
for i, ele in enumerate(sort_idx):
    # imgname está con ruta, obtengo solo últimos dos niveles para la ruta relativa
    imgname_rel = '/'.join(imgnames[ele].split('/')[-2:])
    img = cv2.imread(imgname_rel)
    if img is None:
        print(f"Error leyendo imagen: {imgname_rel}")
        continue
    img = cv2.resize(img, (W, H))
    canvas[int(i / NW) * H: (int(i / NW) + 1) * H, (i % NW) * W: ((i % NW) + 1) * W, :] = img

plt.figure(figsize=(20, 20))
print("Magnitudes ordenadas:", [float(f"{mags[idx_]:.2f}") for idx_ in sort_idx])
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Crear matriz de similitud (producto punto entre vectores normalizados)
feats = np.array([img_2_feats[imgnames[ele]] for ele in sort_idx])
sim_mat = np.dot(feats, feats.T)

# Mostrar matriz de similitud como heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(sim_mat, cmap="PuRd", annot=True, ax=ax)
plt.show()
