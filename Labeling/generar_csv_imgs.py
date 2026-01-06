import os
import csv


def generar_csv_archivos(origin_dir_files, output_csv_path, file_extension='.png', mode='full'):
    
    imagenes_paths = []

    for subdir, dirs, files in os.walk(spectrogram_dir):

        # Ordenar carpetas alfabéticamente
        dirs.sort()

        # Ignorar carpetas que empiezan por número
        base_dir = os.path.basename(subdir)
        if base_dir and base_dir[0].isdigit():
            continue

        # Ignorar archivos ocultos (.DS_Store)
        files = [f for f in files if not f.startswith('.')]

        # Ordenar archivos por nombre + índice numérico
        for file in sorted(
            files,
            key=lambda x: (x.split('_')[0], int(x.split('_')[-1].split('.')[0]))
        ):
            if file.endswith(file_extension):

                full_path = os.path.join(subdir, file)

                if mode == 'full':
                    imagenes_paths.append(full_path)

                elif mode == 'relative':
                    relative_path = os.path.relpath(full_path, origin_dir_files)
                    imagenes_paths.append(relative_path)

                elif mode == 'name':
                    imagenes_paths.append(file)

                else:
                    raise ValueError("Modo no válido. Usa 'full', 'relative' o 'name'.")

    # Guardar CSV
    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['path'])
        for path in imagenes_paths:
            writer.writerow([path])


# -------------------------
# USO
# -------------------------

spectrogram_dir = '../labeling_files/raw_specs'
csv_output_path = '../labeling_files/paths_imgs_spec_FULL.csv'

generar_csv_archivos(spectrogram_dir,csv_output_path,'.png',mode='full')

print(f"📄 CSV generado: {csv_output_path}")