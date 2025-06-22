import os
import chardet

input_file = r'C:\Users\evamb\OneDrive - Universidad Politécnica de Madrid\Escritorio\biometrics_DTU\biometricDTU2025\img.list'
base_folder = r'C:\Users\evamb\OneDrive - Universidad Politécnica de Madrid\Escritorio\biometrics_DTU\biometricDTU2025'

# Detectar codificación
with open(input_file, 'rb') as f:
    rawdata = f.read()
    result = chardet.detect(rawdata)
encoding = result['encoding']
print(f'Codificación detectada: {encoding}')

# Leer archivo con la codificación detectada
with open(input_file, 'r', encoding=encoding) as f:
    abs_paths = f.readlines()

rel_paths = []
for path in abs_paths:
    path = path.strip()
    rel_path = os.path.relpath(path, base_folder)
    rel_path = rel_path.replace('\\', '/')
    rel_paths.append(rel_path + '\n')

with open(input_file, 'w', encoding=encoding) as f:
    f.writelines(rel_paths)


print(f'Convertidas {len(rel_paths)} rutas absolutas a relativas en "{input_file}".')
