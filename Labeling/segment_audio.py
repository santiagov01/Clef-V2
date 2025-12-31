import os
import argparse
import soundfile as sf
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

def configurar_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def obtener_archivos_ogg(input_dir):
    return sorted(glob(os.path.join(input_dir, "**", "*.ogg"), recursive=True))

def load_audio(path):
    with sf.SoundFile(path) as f:
        audio = f.read()
        samplerate = f.samplerate
    return audio, samplerate

def procesar_audio(audio_path, input_dir, output_dir, segment_duration):
    segmentos_generados = 0
    try:
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        relative_path = os.path.relpath(audio_path, input_dir)
        relative_folder = os.path.dirname(relative_path)

        output_subfolder = os.path.join(output_dir, relative_folder)
        os.makedirs(output_subfolder, exist_ok=True)

        data, samplerate = load_audio(audio_path)
        duration = len(data) / samplerate

        if duration > segment_duration:
            num_segments = int(duration // segment_duration) + 1
            for i in range(num_segments):
                start_sample = int(i * segment_duration * samplerate)
                end_sample = int(min((i + 1) * segment_duration * samplerate, len(data)))
                segment_data = data[start_sample:end_sample]

                if len(segment_data) < segment_duration * samplerate and len(segment_data) > 2*samplerate:
                    num_repeats = int(np.ceil((segment_duration * samplerate) / len(segment_data)))
                    segment_data = np.tile(segment_data, num_repeats)[:int(segment_duration * samplerate)]
                    logging.info(f"Extendiendo audio: {audio_path}")

                if len(segment_data) > 0 and len(segment_data) > 2*samplerate:
                    segment_path = os.path.join(output_subfolder, f"{filename}_{i}.ogg")
                    sf.write(segment_path, segment_data, samplerate)
                    segmentos_generados += 1

        else:
            segment_data = data
            if len(segment_data) < segment_duration * samplerate and len(segment_data) > 2*samplerate:
                num_repeats = int(np.ceil((segment_duration * samplerate) / len(segment_data)))
                segment_data = np.tile(segment_data, num_repeats)[:int(segment_duration * samplerate)]
                logging.info(f"Extendiendo audio: {audio_path}")

            if len(segment_data) > 0 and len(segment_data) > 2*samplerate:
                segment_path = os.path.join(output_subfolder, f"{filename}_0.ogg")
                sf.write(segment_path, segment_data, samplerate)
                segmentos_generados += 1

    except Exception as e:
        logging.error(f"Error procesando {audio_path}: {e}")
        print(f"Error procesando {audio_path}: {e}")
    
    return segmentos_generados

def segmentar_audios(input_dir, output_dir, segment_duration=5, num_threads=4):
    audio_paths = obtener_archivos_ogg(input_dir)
    total_segmentos = 0

    print(f"Procesando {len(audio_paths)} archivos con {num_threads} hilos...")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(procesar_audio, path, input_dir, output_dir, segment_duration) for path in audio_paths]
        for future in as_completed(futures):
            total_segmentos += future.result()

    print(f"\nTotal de segmentos generados: {total_segmentos}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentador de audios en clips de duración fija con estructura de carpetas")
    parser.add_argument("--input", type=str, default="raw_audios/", help="Directorio raíz de entrada con archivos de audio")
    parser.add_argument("--output", type=str, default="audios/", help="Directorio raíz de salida donde se guardarán los segmentos")
    parser.add_argument("--duration", type=int, default=5, help="Duración de cada segmento en segundos (default: 5)")
    parser.add_argument("--threads", type=int, default=4, help="Número de hilos para procesamiento paralelo (default: 4)")
    parser.add_argument("--log", type=str, default="Segment_Audio/segmentador.log", help="Ruta del archivo de log (default: segmentador.log)")
    args = parser.parse_args()

    configurar_logging(args.log)
    segmentar_audios(args.input, args.output, segment_duration=args.duration, num_threads=args.threads)

# ======================= EXECUTION========================
# DEFAULT EXECUTION 
# python Audio_Processing/segment.py --threads 4

# default Args
# --input raw_audios/ 
# --output audios/ 
# --duration 5 
# --threads 4
# --log Segment_Audio/segmentador.log
