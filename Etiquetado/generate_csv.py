import os
import pandas as pd

def save_paths_to_csv(directory, output_file):
    """
    Save all file paths in the given directory to a CSV file.

    :param directory: Directory to search for files.
    :param output_file: Output CSV file path.
    """
    # Get all file paths in the directory
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(file_paths, columns=['rutas'])
    df.to_csv(output_file, index=False)
    print(f"Saved {len(file_paths)} file paths to {output_file}") 

save_paths_to_csv(R'D:\birdclef\Birdclef CLEAN\Data\raw_specs', 'paths_imgs_spec_FULL.csv')