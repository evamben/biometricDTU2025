import os

dir_path = 'datasets/Casia-Fasd/test_img/test_img/color/'
output_file = 'img.list'

with open(output_file, 'w') as f:
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.jpg'):
                full_path = os.path.join(root, file)
                f.write(full_path + '\n')

print(f'File {output_file} created with the image paths.')
