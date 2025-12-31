import os
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import time

# Generar el CSV de embeddings BirdNET sin solapamiento

def load_embedding(path, agg="mean"):
    """Lee el embedding, agregando si hay más de una línea en el archivo."""
    embeddings = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            emb = list(map(float, parts[2].split(",")))
            embeddings.append(emb)

    if not embeddings:
        raise ValueError(f"No se pudo extraer ningún embedding válido de: {path}")

    emb_array = np.array(embeddings)
    if agg == 'mean':
        aggregated = np.mean(emb_array, axis=0)
    elif agg == 'max':
        aggregated = np.max(emb_array, axis=0)
    else:
        raise ValueError(f"Agregación no soportada: {agg}")
    
    return list(map(str, aggregated))  # conservar decimales como strings

def parse_filename(path):
    """Extrae info desde el path completo."""
    file = os.path.basename(path)
    label = os.path.basename(os.path.dirname(path))
    audio_id = file.split(".")[0].rsplit("_", 1)[0]
    chunk_index = int(file.split("_")[1].split(".")[0])
    return label, audio_id, chunk_index

def generar_csv_noverlap(input_dir, output_csv, chunk_size=3, num_threads=4, species="all", agg="mean"):
    max_threads = min(num_threads, os.cpu_count())
    all_txt_files = []

    species_to_process = [species] if species != "all" else os.listdir(input_dir)

    for label_folder in species_to_process:
        folder_path = os.path.join(input_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".birdnet.embeddings.txt"):
                all_txt_files.append(os.path.join(folder_path, file))


    audio_chunks = defaultdict(list)

    print(f"Cargando archivos en paralelo con {max_threads} hilos usando agregación: {agg}")
    start_time = time.time()

    def process_file(path):
        emb = load_embedding(path, agg)
        label, audio_id, chunk_idx = parse_filename(path)
        return (label, audio_id, chunk_idx, emb)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_file, path): path for path in all_txt_files}
        for future in as_completed(futures):
            path = futures[future]
            print(f"Procesando {path}")
            try:
                label, audio_id, chunk_idx, emb = future.result()
                audio_chunks[(label, audio_id)].append((chunk_idx, emb))
            except Exception as e:
                print(f"Error en {path}: {e}")

    print("Procesando agrupaciones sin solapamiento...")
    rows = []

    for (label, audio_id), chunks in audio_chunks.items():
        chunks.sort(key=lambda x: x[0])
        embeddings = [e[1] for e in chunks]

        i = 0
        while i < len(embeddings):
            group = embeddings[i:i+chunk_size]
            if len(group) < chunk_size:
                group += [group[-1]] * (chunk_size - len(group))
            group_id = i // chunk_size
            row_id = audio_id
            concatenated = sum(group, [])
            row = [row_id, str(group_id), label] + concatenated
            rows.append(row)
            i += chunk_size

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    num_features = chunk_size * 1024
    columns = ["row_id", "group", "label"] + [f"emb_{i}" for i in range(num_features)]
    df = pd.DataFrame(rows, columns=columns, dtype=str)
    df.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    print(f"CSV guardado en {output_csv} con {len(rows)} filas.")
    print(f"Tiempo total: {elapsed:.2f} segundos.")

# ======================= EJECUCIÓN ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador CSV sin solapamiento desde embeddings BirdNET")
    parser.add_argument("--input", type=str, default="embeddings", help="Directorio raíz de entrada con carpetas por clase")
    parser.add_argument("--output", type=str, default="embeddings_csv/embeddings_MT_noverlap.csv", help="Ruta del archivo CSV de salida")
    parser.add_argument("--chunks", type=int, default=3, help="Número de embeddings a concatenar por fila (sin solape)")
    parser.add_argument("--threads", type=int, default=4, help="Número de hilos para procesamiento paralelo")
    parser.add_argument("--species", type=str, default="all", help="Nombre de la especie a procesar (subcarpeta). Usa 'all' para procesar todas.")
    parser.add_argument("--agg", type=str, default="mean", choices=["mean", "max"], help="Método de agregación para múltiples líneas en un archivo de embedding")
    args = parser.parse_args()

    generar_csv_noverlap(
        args.input,
        args.output,
        chunk_size=args.chunks,
        num_threads=args.threads,
        species=args.species,
        agg=args.agg
    )

# ======================= EXECUTION ========================

# DEFAULT EXECUTION 
# python embed2csv/embed_MT_P_NOV.py --threads 4 

# CSV ESPECIES
# python embed2csv/embed_MT_P_NOV.py --output embeddings_csv/species/embeddings_yeofly1.csv --chunks 1 --threads 4 --species yeofly1

# default Args
# --input embeddings 
# --output embeddings_csv/embeddings_MT_noverlap.csv 
# --chunks 3 
# --threads 4
# --species all
# --agg mean