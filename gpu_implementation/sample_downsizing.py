import os
import glob
import tifffile
import numpy as np
from pathlib import Path
from tqdm import tqdm

def downsample_tiff_folder(input_folder, output_folder, downsample_factor=4):
    """
    Downsample all TIFF files in a folder by a given factor
    
    Parameters:
    -----------
    input_folder : str
        Path to folder containing original TIFF files
    output_folder : str
        Path to folder where downsampled TIFFs will be saved
    downsample_factor : int
        Factor to downsample (4 = quarter size)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all TIFF files
    tiff_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tiff_files = []
    for pattern in tiff_patterns:
        tiff_files.extend(glob.glob(os.path.join(input_folder, pattern)))
    
    if len(tiff_files) == 0:
        print(f"No TIFF files found in {input_folder}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files")
    print(f"Downsampling by factor of {downsample_factor}")
    print(f"Output folder: {output_folder}")
    
    # Process each file
    for filepath in tqdm(tiff_files, desc="Downsampling TIFFs"):
        try:
            # Load TIFF
            data = tifffile.imread(filepath)
            
            # Handle different dimensions
            if data.ndim == 3 and data.shape[0] == 1:
                data = data.squeeze(0)
            
            # Downsample using array slicing
            if data.ndim == 2:
                downsampled = data[::downsample_factor, ::downsample_factor]
            elif data.ndim == 3:
                # Multi-page TIFF
                downsampled = data[:, ::downsample_factor, ::downsample_factor]
            else:
                print(f"Skipping {filepath}: unexpected dimensions {data.shape}")
                continue
            
            # Create output filename
            filename = os.path.basename(filepath)
            output_path = os.path.join(output_folder, filename)
            
            # Save downsampled TIFF
            tifffile.imwrite(output_path, downsampled)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print(f"\n✓ Completed! Downsampled files saved to {output_folder}")


if __name__ == "__main__":
    # Configuration
    INPUT_FOLDER = "data/20240530_ITRI/slices"  # Change this to your input folder
    OUTPUT_FOLDER = "data/20240530_ITRI_downsampled_4x/slices"  # Change this to your output folder
    DOWNSAMPLE_FACTOR = 4
    
    # Run downsampling
    downsample_tiff_folder(INPUT_FOLDER, OUTPUT_FOLDER, DOWNSAMPLE_FACTOR)