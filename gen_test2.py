# generar_test2_csv.py

def generar_csv_desde_feat_list(feat_list_path, salida_csv):
    with open(feat_list_path, 'r') as f:
        lines = f.readlines()

    with open(salida_csv, 'w') as f_out:
        f_out.write("photo_path,photo_label\n")
        for line in lines:
            partes = line.strip().split()
            if len(partes) >= 1:
                photo_path = partes[0]
                label = 1 if "real" in photo_path.lower() else 0
                f_out.write(f"{photo_path},{label}\n")

if __name__ == "__main__":
    feat_list_path = "feat.list"         # Cambia esto si tu archivo tiene otro nombre
    salida_csv = "test2.csv"
    generar_csv_desde_feat_list(feat_list_path, salida_csv)
    print(f"Archivo '{salida_csv}' generado correctamente.")
