import numpy as np
import os
import time
import logging
import json
from PIL import Image
from tqdm import tqdm
import traceback
import sys
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List, Union
import astra
import pickle
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_raw(path, header_bytes, endianness, resolution, bit_depth) -> np.ndarray:
    """Load RAW binary file with optional header skip and endianness control"""
    base_dtype = np.dtype(bit_depth)
    
    if endianness == "little":
        dtype = base_dtype.newbyteorder('<')
    elif endianness == "big":
        dtype = base_dtype.newbyteorder('>')
    else:
        dtype = base_dtype
    
    height, width = resolution  # Example resolution, should be parameterized
    expected_data_size = height * width * dtype.itemsize
    expected_total_size = header_bytes + expected_data_size
    actual_size = os.path.getsize(path)
    
    if actual_size < expected_total_size:
        logger.warning(f"File {os.path.basename(path)} smaller than expected, recalculating header")
        header_bytes = actual_size - expected_data_size
        if header_bytes < 0:
            raise ValueError(f"File size {actual_size} incompatible with resolution {height}×{width}")
        logger.info(f"Adjusted header to {header_bytes} bytes")
    
    with open(path, "rb") as f:
        if header_bytes > 0:
            f.seek(header_bytes)
        data = np.frombuffer(f.read(), dtype=dtype)
    
    try:
        data = data.reshape((height, width))
    except ValueError:
        logger.warning(f"Reshape failed, trying transpose")
        data = data.reshape((width, height)).T
    
    return data.astype(np.dtype(bit_depth))

def show_image(image: np.ndarray, title: str = "Image") -> None:
    """Display an image using PIL"""
    img = Image.fromarray(image.astype(np.dtype(image.dtype)))
    img.show(title=title)

def save_image(image: np.ndarray, bit_depth: str, path: str) -> None:
    """Save an image to the specified path using PIL"""
    img = Image.fromarray(image.astype(np.dtype(image.dtype)))
    img.save(path)
    logger.info(f"Image saved to {path}")

if __name__ == "__main__":
    try:
        # Example usage
        dataset_input_path = "data/20251119_Tako_SiC/slices/"
        file_collection = os.listdir(dataset_input_path)
        file_name = ""  # Example file name, should be parameterized
        for file_name in file_collection:
            # print(file_name)
            if file_name.endswith(".RAW") or file_name.endswith(".raw"):
                file_name = file_name[:-4]  # Remove .raw extension
                raw_file_path = os.path.join(dataset_input_path, file_name + ".raw")
                
                image_output_path = os.path.join("data/20251119_Tako_SiC/converted/", file_name + ".png")
                header_size = 0  # Example header size, should be parameterized
                endianness = "little"  # Example endianness, should be parameterized
                
                image = load_raw(raw_file_path, header_size, endianness, (3072, 3072), "uint16")
                logger.info(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")
                # show_image(image, "int16")
                save_image(image, image.dtype, image_output_path)
                # quit()

    except Exception as e:
        logger.error(f"Failed to load RAW file: {e}")
        traceback.print_exc()
        sys.exit(1)