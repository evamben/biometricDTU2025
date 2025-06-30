import os

input_folder = "datasets/Casia-Fasd/test_img/test_img/color"
output_list_path = "Magface_v2/preprocessing_v1/Casia_Fasd/test/img.list"

with open(output_list_path, "r") as f:
    lines = f.readlines()

fixed_lines = [line.replace("\\", "/") for line in lines]

with open("Magface_v2/preprocessing_v1/Casia_Fasd/img_fixed.list", "w") as f:
    f.writelines(fixed_lines)

print("Fixed list saved as dataset/img_fixed.list")
