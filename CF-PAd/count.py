import os

def count_images_in_folder(folder, exts={".jpg", ".jpeg", ".png"}):
    count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(tuple(exts)):
                count += 1
    return count

# Paths (modifica si es necesario)
casia_color_path = "datasets/Casia-Fasd/test_img/test_img/color"
lcc_real_path = "datasets/LCC_FASD/LCC_FASD_development/real"
lcc_spoof_path = "datasets/LCC_FASD/LCC_FASD_development/spoof"

# Contar imágenes
casia_color_count = count_images_in_folder(casia_color_path)
lcc_real_count = count_images_in_folder(lcc_real_path)
lcc_spoof_count = count_images_in_folder(lcc_spoof_path)

# Mostrar resultados
print(f"CASIA (color test images): {casia_color_count}")
print(f"LCC (real): {lcc_real_count}")
print(f"LCC (spoof): {lcc_spoof_count}")
print(f"LCC (total): {lcc_real_count + lcc_spoof_count}")
