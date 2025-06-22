import os

# Ruta al directorio donde están las imágenes
dir_path = 'datasets/Casia-Fasd/test_img/test_img/color/'

# Archivo de salida
output_file = 'img.list'

with open(output_file, 'w') as f:
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.jpg'):
                full_path = os.path.join(root, file)
                f.write(full_path + '\n')

print(f'Archivo {output_file} creado con las rutas de las imágenes.')
