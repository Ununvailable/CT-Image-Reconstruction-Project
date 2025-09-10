import numpy as np
import os
import glob
from pathlib import Path
from PIL import Image
import time

from joblib import Parallel, delayed

import multiprocessing

def padImage(img):
    N0, N1 = img.size
    lenDiag = int(np.ceil(np.sqrt(N0**2 + N1**2)))
    imgPad = Image.new('L', (lenDiag, lenDiag))
    c0, c1 = int(round((lenDiag - N0) / 2)), int(round((lenDiag - N1) / 2))
    imgPad.paste(img, (c0, c1))
    return imgPad, c0, c1


def _single_proj(img_arr, angle_deg):
    # rotate using NumPy/OpenCV-like transformation
    img_rot = Image.fromarray(img_arr).rotate(90 - angle_deg, resample=Image.BICUBIC)
    # print(f"Single projection: {angle_deg} degree")
    return np.sum(np.array(img_rot), axis=0)


def getProj(img, theta, n_jobs=None, show_progress=False):
    """
    Parallel Radon projection.
    """
    numAngles = len(theta)
    img_arr = np.array(img)

    # Parallel processing
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_single_proj)(img_arr, theta[n]) for n in range(numAngles)
    )

    sinogram = np.stack(results, axis=1)
    return sinogram


if __name__ == '__main__':
    cpu_cores = multiprocessing.cpu_count()
    print(f"Detected CPU cores: {cpu_cores}")

    # Define folders
    input_folder = 'data/phantoms/'
    sinogram_folder = 'data/sinogram/'
    
    # Create sinogram folder
    os.makedirs(sinogram_folder, exist_ok=True)
    
    # Get all PNG files
    image_files = glob.glob(os.path.join(input_folder, '*.png'))
    
    if not image_files:
        print(f"No PNG files found in {input_folder}")
        exit()
    
    print(f"Found {len(image_files)} PNG files to process")
    
    # Processing parameters
    dTheta = 0.1
    theta = np.arange(0, 361, dTheta)
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"\n--- Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)} ---")
        
        try:
            # Load and prepare image
            myImg = Image.open(image_path).convert('L')
            myImgPad, c0, c1 = padImage(myImg)
            
            start_time = time.perf_counter()
            
            print('Getting projections...')
            mySino = getProj(myImgPad, theta, n_jobs=cpu_cores)

            end_time = time.perf_counter()
            print(f"Forward projection time: {end_time - start_time:.6f} seconds")
            
            # Save sinogram and metadata
            input_filename = Path(image_path).stem
            sino_filename = f"{input_filename}_sinogram_{dTheta}.npz"
            sino_path = os.path.join(sinogram_folder, sino_filename)
            
            # Save with metadata for reconstruction
            np.savez(sino_path, 
                    sinogram=mySino,
                    theta=theta,
                    original_size=(myImg.size[0], myImg.size[1]),
                    pad_offset=(c0, c1))
            
            print(f"Saved sinogram: {sino_path}")
            print(f"Sinogram shape: {mySino.shape}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    print(f"\n--- Forward projection complete! ---")
    print(f"Processed {len(image_files)} images")
    print(f"Sinograms saved in: {sinogram_folder}")