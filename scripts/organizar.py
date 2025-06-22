import os
import shutil

# Carpeta destino organizada
output_base_dir = 'quality_split_images'
os.makedirs(output_base_dir, exist_ok=True)

# Las categorías que vamos a procesar
qualities = ['low_quality', 'medium_quality', 'high_quality']

# Cómo determinar si es real o fake
def get_label(img_path):
    img_path_lower = img_path.lower()
    if 'real' in img_path_lower:
        return 'real'
    elif 'fake' in img_path_lower:
        return 'fake'
    else:
        raise ValueError(f"No se pudo determinar si es real o fake: {img_path}")

# Procesar cada archivo de calidad
for quality in qualities:
    txt_path = os.path.join('quality_splits', f'{quality}.txt')
    
    with open(txt_path, 'r') as f:
        img_paths = [line.strip() for line in f.readlines()]
    
    for img_path in img_paths:
        if not os.path.isfile(img_path):
            print(f"⚠️  Imagen no encontrada: {img_path}")
            continue

        try:
            label = get_label(img_path)
        except ValueError as e:
            print(f"⚠️  {e}")
            continue

        # Crear la carpeta de destino
        dest_dir = os.path.join(output_base_dir, quality, label)
        os.makedirs(dest_dir, exist_ok=True)

        # Nombre base del archivo (sin directorios)
        img_filename = os.path.basename(img_path)
        dest_path = os.path.join(dest_dir, img_filename)

        try:
            shutil.copy(img_path, dest_path)
            print(f"✅ Copiado: {img_path} -> {dest_path}")
        except Exception as e:
            print(f"❌ Error copiando {img_path}: {e}")
