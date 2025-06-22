import csv

input_csv = 'test.csv'
output_csv = 'test_2.csv'

def fix_path(path):
    # Cambia '\' por '/' y elimina la repetición 'train_img/train_img'
    path = path.replace('\\', '/')
    return path

with open(input_csv, 'r', newline='', encoding='utf-8') as infile, \
     open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if len(row) > 0:
            original_path = row[0]
            fixed_path = fix_path(original_path)
            row[0] = fixed_path
        writer.writerow(row)

print(f"Archivo corregido guardado como {output_csv}")
