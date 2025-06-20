import os

input_folder = "datasets/Casia-Fasd/train_img/train_img/color"
output_list_path = "preprocessing_v1/Casia_Fasd/img.list"

with open(output_list_path, "r") as f:
    lines = f.readlines()

fixed_lines = [line.replace("\\", "/") for line in lines]

with open("preprocessing_v1/Casia_Fasd/img_fixed.list", "w") as f:
    f.writelines(fixed_lines)

print("✅ Fixed list saved as dataset/img_fixed.list")
