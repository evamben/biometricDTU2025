import os
import shutil
import random

def split_train_val_flat(train_dir, val_dir, val_ratio=0.1, move_files=False):
    """
    Divide un conjunto de imágenes todas en la misma carpeta en train + val,
    usando val_ratio para sacar porcentaje para validación.
    Asume que la clase está implícita en el nombre del archivo.
    """

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    images = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    random.shuffle(images)

    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]

    for img in val_images:
        src = os.path.join(train_dir, img)
        dst = os.path.join(val_dir, img)

        if move_files:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)

    print(f"Copiadas {val_count} imágenes para validación.")

if __name__ == "__main__":
    train_folder = r"Magface_v2\datasets\Casia-Fasd\train_img\train_img\color"
    val_folder = r"Magface_v2\datasets\Casia-Fasd\val_img\color"
    split_train_val_flat(train_folder, val_folder, val_ratio=0.1, move_files=False)
python train.py  --train_dir Magface_v2/datasets/Casia-Fasd/train_img/train_img   --val_dir Magface_v2/datasets/Casia-Fasd/val_img   --batch_size 32  --epochs 20  --lr 0.001
