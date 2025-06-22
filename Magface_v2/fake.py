import os

# Directorio donde están las listas por calidad
input_dir = 'quality_splits'

# Archivos de listas para 5 rangos
quality_files = [
    'very_low_quality.txt',
    'low_quality.txt',
    'medium_quality.txt',
    'high_quality.txt',
    'very_high_quality.txt'
]

def count_real_fake(file_path):
    real_count = 0
    fake_count = 0
    with open(file_path, 'r') as f:
        for line in f:
            img_path = line.strip()
            # Ajusta esta lógica según cómo distinguir real/fake en tu dataset
            if 'real' in img_path.lower():
                real_count += 1
            elif 'fake' in img_path.lower():
                fake_count += 1
            else:
                # Si no puedes determinar, puedes contar aparte o ignorar
                pass
    return real_count, fake_count

for qfile in quality_files:
    path = os.path.join(input_dir, qfile)
    real, fake = count_real_fake(path)
    print(f"{qfile} -> Reales: {real}, Fakes: {fake}")
