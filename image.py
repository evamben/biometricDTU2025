from PIL import Image


print(img.size)
import os
from PIL import Image

def validar_imagenes(root_dir, expected_size=(256, 256), expected_mode='RGB'):
    errores = []
    
    for carpeta in os.listdir(root_dir):
        carpeta_path = os.path.join(root_dir, carpeta)
        if not os.path.isdir(carpeta_path):
            continue

        for archivo in os.listdir(carpeta_path):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                ruta_img = os.path.join(carpeta_path, archivo)
                try:
                    with Image.open(ruta_img) as img:
                        if img.size != expected_size or img.mode != expected_mode:
                            errores.append((ruta_img, img.size, img.mode))
                except Exception as e:
                    errores.append((ruta_img, 'Error al abrir', str(e)))

    if errores:
        print("🔴 Se encontraron imágenes que no cumplen con el formato esperado:")
        for ruta, size, mode in errores:
            print(f"- {ruta} → Tamaño: {size}, Modo: {mode}")
    else:
        print("✅ Todas las imágenes tienen tamaño 256x256 y son RGB.")

# 👉 Cambia esta ruta por la tuya
ruta_dataset = 'datasets/Casia-Fasd/train_img/train_img/color'
validar_imagenes(ruta_dataset)
