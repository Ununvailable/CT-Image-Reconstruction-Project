import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.fftpack import fft, fftshift, ifft
import time
import os
import glob
from pathlib import Path

# from numba import cuda
import cupy as cp

# Reuse CPU projection from filtbackproj_multicore_full_parallelization.py
from filtbackproj_multicore_full_parallelization import getProj as getProj_cpu
import multiprocessing


def dummyImg(size0, size1):
    M = np.zeros((size0, size1))
    M[190:210, :] = 255
    M[:, 210:230] = 255
    dumImg = Image.fromarray(M.astype('uint8'))
    return dumImg


def padImage(img):
    N0, N1 = img.size
    lenDiag = int(np.ceil(np.sqrt(N0**2 + N1**2)))
    imgPad = Image.new('L', (lenDiag, lenDiag))
    c0, c1 = int(round((lenDiag - N0) / 2)), int(round((lenDiag - N1) / 2))
    imgPad.paste(img, (c0, c1))
    return imgPad, c0, c1

def getProj(img, theta, plot_progress, n_jobs=None):
    numAngles = len(theta)
    sinogram = np.zeros((img.size[0],numAngles)) # (y, x)

    # Set up plotting
    if plot_progress == True:
        plt.ion()
        fig1, (ax1, ax2) = plt.subplots(1,2)
        im1 = ax1.imshow(img, cmap='gray')
        ax1.set_title('<-- Sum')
        im2 = ax2.imshow(sinogram, extent=[theta[0],theta[-1], img.size[0]-1, 0],
                        cmap='gray', aspect='auto')
        ax2.set_xlabel('Angle (deg)')
        ax2.set_title('Sinogram')
        plt.show()

        # Get projections an dplot
        for n in range(numAngles):
            rotImgObj = img.rotate(90-theta[n], resample=Image.BICUBIC)
            im1.set_data(rotImgObj)
            sinogram[:,n] = np.sum(rotImgObj, axis=0) 
            
            im2.set_data(Image.fromarray((sinogram-np.min(sinogram))/np.ptp(sinogram)*255))
            fig1.canvas.draw() 
            fig1.canvas.flush_events() 
        plt.ioff() 
    return sinogram

def projFilter_gpu(sino_np: np.ndarray) -> np.ndarray:
    """
    GPU version of projFilter using CuPy FFT along the projection axis (axis=0).
    Accepts NumPy, returns NumPy.
    """
    sino = cp.asarray(sino_np, dtype=cp.float32)                # (projLen, numAngles)

    projLen, numAngles = sino.shape
    a = 0.5
    step = 2 * np.pi / projLen

    # w like your arange2(), ensured length == projLen
    w = cp.arange(-cp.pi, cp.pi, step, dtype=cp.float32)
    if w.size < projLen:
        w = cp.concatenate([w, w[-1:] + step])

    rn1 = cp.abs(2 / a * cp.sin(a * w / 2))
    rn2 = cp.sin(a * w / 2) / (a * w / 2)
    rn2 = cp.nan_to_num(rn2, nan=1.0)                            # handle 0/0 at w=0
    r = rn1 * (rn2 ** 2)
    filt = cp.fft.fftshift(r).astype(cp.complex64)               # (projLen,)

    projfft = cp.fft.fft(sino, axis=0)                           # (projLen, numAngles)
    filtProj = projfft * filt[:, None]
    filtSino = cp.real(cp.fft.ifft(filtProj, axis=0)).astype(cp.float32)

    return cp.asnumpy(filtSino)

_backproj_src = r"""
extern "C" __global__
void backproj_kernel(const float* __restrict__ sino,     // (N, A) flattened row-major: idx = s*A + a
                     const float* __restrict__ sin_th,   // (A,)
                     const float* __restrict__ cos_th,   // (A,)
                     float* __restrict__ out,            // (N, N) flattened row-major: idx = y*N + x
                     const int N,
                     const int A)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;  // [0, N)
    int y = blockDim.y * blockIdx.y + threadIdx.y;  // [0, N)
    if (x >= N || y >= N) return;

    // Centered coordinates (match Python: np.arange(N) - N/2)
    float xf = (float)x - 0.5f * (float)N;
    float yf = (float)y - 0.5f * (float)N;

    float acc = 0.0f;
    for (int a = 0; a < A; ++a) {
        float s = xf * sin_th[a] - yf * cos_th[a] + 0.5f * (float)N;
        int si = __float2int_rn(s);  // round to nearest (like np.round)
        if (0 <= si && si < N) {
            acc += sino[si * A + a];
        }
    }
    out[y * N + x] = acc;
}
""";

_backproj_kernel = cp.RawKernel(_backproj_src, "backproj_kernel")

def backProj_gpu(sinogram_np: np.ndarray, theta_deg_np: np.ndarray) -> np.ndarray:
    """
    GPU backprojection. Accepts NumPy sinogram (N, A) and theta in degrees.
    Returns NumPy (N, N) image (flipped to match your CPU version).
    """
    sino = cp.asarray(sinogram_np, dtype=cp.float32)
    N, A = sino.shape

    theta = cp.asarray(theta_deg_np, dtype=cp.float32) * (cp.pi / 180.0)
    sin_th = cp.sin(theta).astype(cp.float32)
    cos_th = cp.cos(theta).astype(cp.float32)

    out = cp.zeros((N, N), dtype=cp.float32)

    # Launch: 2D grid over pixels
    block = (16, 16, 1)
    grid = ((N + block[0] - 1) // block[0],
            (N + block[1] - 1) // block[1],
            1)

    _backproj_kernel(grid, block,
                     (sino.ravel(), sin_th, cos_th, out.ravel(),
                      np.int32(N), np.int32(A)))

    # Match your original post-processing: vertical flip
    backprojArray = cp.flipud(out)
    return cp.asnumpy(backprojArray)


if __name__ == '__main__':
    cpu_cores = multiprocessing.cpu_count()
    print(f"Detected CPU cores: {cpu_cores}")

    # Define input and output folders
    input_folder = 'data/phantoms/'  # Change this to your input folder
    output_folder = 'data/reconstructed/'  # Change this to your output folder
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all PNG files in the folder
    image_files = glob.glob(os.path.join(input_folder, '*.png'))
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        exit()
    
    print(f"Found {len(image_files)} image files to process")
    
    # Processing parameters
    dTheta = 1
    theta = np.arange(0, 361, dTheta)

    # GPU initialization and kernel compilation (warm-up)
    print("Initializing GPU and compiling kernels...")
    dummy_sino = cp.ones((256, 180), dtype=cp.float32)
    dummy_theta = cp.linspace(0, 180, 180, dtype=cp.float32)
    
    # Warm up projFilter_gpu
    _ = projFilter_gpu(cp.asnumpy(dummy_sino))
    
    # Warm up backProj_gpu (compiles CUDA kernel)
    _ = backProj_gpu(cp.asnumpy(dummy_sino), cp.asnumpy(dummy_theta))
    
    print("GPU initialization complete.")

    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"\n--- Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)} ---")
        
        try:
            # Load and prepare image
            myImg = Image.open(image_path).convert('L')
            myImgPad, c0, c1 = padImage(myImg)
            
            print('Getting projections (sequential)...')
            mySino = getProj_cpu(myImgPad, theta, n_jobs=cpu_cores)
            
            start_time = time.perf_counter()
            
            print('Filtering (parallel)...')
            filtSino = projFilter_gpu(mySino)
            
            print('Performing backprojection (parallel)...')
            recon = backProj_gpu(filtSino, theta)
            
            end_time = time.perf_counter()
            print(f"Processing time: {end_time - start_time:.6f} seconds")
            
            # Post-process and save
            recon2 = np.round((recon - np.min(recon)) / np.ptp(recon) * 255)
            reconImg = Image.fromarray(recon2.astype('uint8'))
            n0, n1 = myImg.size
            reconImg = reconImg.crop((c0, c1, c0 + n0, c1 + n1))
            
            # Generate output filename
            input_filename = Path(image_path).stem
            output_filename = f"{input_filename}_reconstructed.png"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save reconstructed image
            reconImg.save(output_path)
            print(f"Saved reconstructed image: {output_path}")
            
            # Optional: Save comparison figure for each image
            save_comparison = False  # Set to False if you don't want individual comparison plots
            if save_comparison:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
                ax1.imshow(myImg, cmap='gray')
                ax1.set_title('Original Image')
                ax1.axis('off')
                
                ax2.imshow(reconImg, cmap='gray')
                ax2.set_title('Reconstructed Image')
                ax2.axis('off')
                
                ax3.imshow(ImageChops.difference(myImg, reconImg), cmap='gray')
                ax3.set_title('Error')
                ax3.axis('off')
                
                plt.tight_layout()
                comparison_path = os.path.join(output_folder, f"{input_filename}_comparison.png")
                plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
                plt.close()  # Close to free memory
                print(f"Saved comparison plot: {comparison_path}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    print(f"\n--- Batch processing complete! ---")
    print(f"Processed {len(image_files)} images")
    print(f"Results saved in: {output_folder}")