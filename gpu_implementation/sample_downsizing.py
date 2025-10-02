import os
import glob
import tifffile
import numpy as np
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm

def downsample_tiff_folder(input_folder, output_folder, downsample_factor=4):
    """
    Downsample all TIFF files in a folder by a given factor with consistent output dimensions
    
    Parameters:
    -----------
    input_folder : str
        Path to folder containing original TIFF files
    output_folder : str
        Path to folder where downsampled TIFFs will be saved
    downsample_factor : int
        Factor to downsample (4 = quarter size)
    """
    os.makedirs(output_folder, exist_ok=True)
    
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
    
    for filepath in tqdm(tiff_files, desc="Downsampling TIFFs"):
        try:
            data = tifffile.imread(filepath)
            
            if data.ndim == 3 and data.shape[0] == 1:
                data = data.squeeze(0)
            
            if data.ndim == 2:
                h, w = data.shape
                new_h, new_w = h // downsample_factor, w // downsample_factor
                downsampled = resize(data, (new_h, new_w), order=1, 
                                    preserve_range=True, anti_aliasing=True).astype(data.dtype)
            elif data.ndim == 3:
                d, h, w = data.shape
                new_h, new_w = h // downsample_factor, w // downsample_factor
                downsampled = resize(data, (d, new_h, new_w), order=1, 
                                    preserve_range=True, anti_aliasing=True).astype(data.dtype)
            else:
                print(f"Skipping {filepath}: unexpected dimensions {data.shape}")
                continue
            
            filename = os.path.basename(filepath)
            output_path = os.path.join(output_folder, filename)
            tifffile.imwrite(output_path, downsampled)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print(f"\n✓ Completed! Downsampled files saved to {output_folder}")

if __name__ == "__main__":
    INPUT_FOLDER = "data/20240530_ITRI/slices"
    OUTPUT_FOLDER = "data/20240530_ITRI_downsampled_4x/slices"
    DOWNSAMPLE_FACTOR = 4
    
    downsample_tiff_folder(INPUT_FOLDER, OUTPUT_FOLDER, DOWNSAMPLE_FACTOR)