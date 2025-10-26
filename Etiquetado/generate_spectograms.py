from joblib import Parallel, delayed
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging


# input_dir = R'D:\birdclef\Birdclef CLEAN\borrar\audios'
# output_dir = R'D:\birdclef\Birdclef CLEAN\borrar\specs'
sr = 32000
hop_length = 1024
n_mels = 128

def configurar_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def procesar_audio(file_path, input_dir, output_dir):
    try:
        relative_path = os.path.relpath(os.path.dirname(file_path), input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        output_file = os.path.join(output_subdir, os.path.basename(file_path).replace('.ogg', '.png'))
        
        if os.path.exists(output_file):
            print(f"El archivo {output_file} ya existe, omitiendo...")
            return f"{file_path} ya existe."

        y, sr_actual = librosa.load(file_path, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr_actual, n_mels=n_mels, hop_length=hop_length)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(5,5))
        librosa.display.specshow(S_dB, sr=sr_actual, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        logging.info(f"Procesado:{output_file}")
        return f"{file_path}"
    except Exception as e:
        logging.error(f"Error procesando {file_path}: {e}")
        return f"{file_path}: {e}"

if __name__ == '__main__':
    print("Iniciando procesamiento de espectrogramas...")

    parser = argparse.ArgumentParser(description="Generador de espectrogramas a partir de archivos de audio")
    parser.add_argument("--input", type=str, default="audios/", help="Directorio raíz de entrada con archivos de audio")
    parser.add_argument("--output", type=str, default="specs/", help="Directorio raíz de salida donde se guardarán los espectrogramas")
    parser.add_argument("--threads", type=int, default=4, help="Número de hilos para procesamiento paralelo (default: 4)")
    parser.add_argument("--log", type=str, default="Segment_Audio/segmentador.log", help="Ruta del archivo de log (default: segmentador.log)")
    args = parser.parse_args()
    configurar_logging(args.log)

    archivos_ogg = [os.path.join(subdir, file)
                    for subdir, _, files in os.walk(args.input)
                    for file in files if file.endswith('.ogg')]

    resultados = Parallel(n_jobs=args.threads)(delayed(procesar_audio)(f, args.input, args.output) for f in archivos_ogg)
    for r in resultados:
        logging.info(r)

# ejemplo de uso
# python Code/Preprocess/crear_espectrogramas.py --input borrar/audios/ --output borrar/specs/ --threads 4 --log borrar/Specs_segm/segmentador.log
