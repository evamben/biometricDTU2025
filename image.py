import os
import glob
import csv

def casia_to_csv():
    data_root = 'datasets/Casia-Fasd'
    output_dir = './casia_csvs/'
    os.makedirs(output_dir, exist_ok=True)

    splits = {
        'train': os.path.join(data_root, 'train_img', 'train_img', 'color'),
        'test': os.path.join(data_root, 'test_img', 'test_img', 'color')
    }

    for split_name, folder in splits.items():
        path_list = glob.glob(os.path.join(folder, '*.jpg'))
        path_list.sort()

        csv_path = os.path.join(output_dir, f'{split_name}.csv')

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Cabecera (opcional)
            writer.writerow(['photo_path', 'photo_label'])

            for path in path_list:
                filename = os.path.basename(path).lower()
                if '_real.jpg' in filename:
                    label = 1
                elif '_fake.jpg' in filename:
                    label = 0
                else:
                    print(f"[!] Nombre ambiguo (sin '_real' o '_fake'): {filename}")
                    continue

                writer.writerow([path, label])

        print(f"[✓] Archivo CSV generado: {csv_path}")

if __name__ == "__main__":
    casia_to_csv()
