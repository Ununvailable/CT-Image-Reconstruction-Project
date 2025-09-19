import numpy as np
import os
import glob
from pathlib import Path
from PIL import Image
import time
import cupy as cp

# GPU functions (projFilter_gpu, backProj_gpu, _backproj_kernel) - keep your existing implementations
def projFilter_gpu(sino_np: np.ndarray) -> np.ndarray:
    """GPU version of projFilter using CuPy FFT along the projection axis (axis=0)."""
    sino = cp.asarray(sino_np, dtype=cp.float32)
    projLen, numAngles = sino.shape
    a = 0.5
    step = 2 * np.pi / projLen

    w = cp.arange(-cp.pi, cp.pi, step, dtype=cp.float32)
    if w.size < projLen:
        w = cp.concatenate([w, w[-1:] + step])

    rn1 = cp.abs(2 / a * cp.sin(a * w / 2))
    rn2 = cp.sin(a * w / 2) / (a * w / 2)
    rn2 = cp.nan_to_num(rn2, nan=1.0)
    r = rn1 * (rn2 ** 2)
    filt = cp.fft.fftshift(r).astype(cp.complex64)

    projfft = cp.fft.fft(sino, axis=0)
    filtProj = projfft * filt[:, None]
    filtSino = cp.real(cp.fft.ifft(filtProj, axis=0)).astype(cp.float32)

    return cp.asnumpy(filtSino)

_backproj_src = r"""
extern "C" __global__
void backproj_kernel(const float* __restrict__ sino,
                     const float* __restrict__ sin_th,
                     const float* __restrict__ cos_th,
                     float* __restrict__ out,
                     const int N,
                     const int A)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= N || y >= N) return;

    float xf = (float)x - 0.5f * (float)N;
    float yf = (float)y - 0.5f * (float)N;

    float acc = 0.0f;
    for (int a = 0; a < A; ++a) {
        float s = xf * sin_th[a] - yf * cos_th[a] + 0.5f * (float)N;
        int si = __float2int_rn(s);
        if (0 <= si && si < N) {
            acc += sino[si * A + a];
        }
    }
    out[y * N + x] = acc;
}
""";

_backproj_kernel = cp.RawKernel(_backproj_src, "backproj_kernel")

def backProj_gpu(sinogram_np: np.ndarray, theta_deg_np: np.ndarray) -> np.ndarray:
    """GPU backprojection."""
    sino = cp.asarray(sinogram_np, dtype=cp.float32)
    N, A = sino.shape

    theta = cp.asarray(theta_deg_np, dtype=cp.float32) * (cp.pi / 180.0)
    sin_th = cp.sin(theta).astype(cp.float32)
    cos_th = cp.cos(theta).astype(cp.float32)

    out = cp.zeros((N, N), dtype=cp.float32)

    block = (16, 16, 1)
    grid = ((N + block[0] - 1) // block[0], (N + block[1] - 1) // block[1], 1)

    _backproj_kernel(grid, block, (sino.ravel(), sin_th, cos_th, out.ravel(),
                      np.int32(N), np.int32(A)))

    backprojArray = cp.flipud(out)
    return cp.asnumpy(backprojArray)

if __name__ == '__main__':
    # Define folders
    sinogram_folder = 'data/sinogram/'
    output_folder = 'data/reconstructed/theta_0p1/'
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all sinogram files
    sino_files = glob.glob(os.path.join(sinogram_folder, '*_sinogram.npz'))
    
    if not sino_files:
        print(f"No sinogram files found in {sinogram_folder}")
        exit()
    
    print(f"Found {len(sino_files)} sinogram files to reconstruct")

    # GPU initialization and kernel compilation (warm-up)
    print("Initializing GPU and compiling kernels...")
    dummy_sino = cp.ones((256, 180), dtype=cp.float32)
    dummy_theta = cp.linspace(0, 180, 180, dtype=cp.float32)

    # Warm up projFilter_gpu
    _ = projFilter_gpu(cp.asnumpy(dummy_sino))
    
    # Warm up backProj_gpu (compiles CUDA kernel)
    _ = backProj_gpu(cp.asnumpy(dummy_sino), cp.asnumpy(dummy_theta))
    
    print("GPU initialization complete.")
    
    # Process each sinogram
    for i, sino_path in enumerate(sino_files):
        print(f"\n--- Processing sinogram {i+1}/{len(sino_files)}: {os.path.basename(sino_path)} ---")
        
        try:
            # Load sinogram data
            data = np.load(sino_path)
            mySino = data['sinogram']
            theta = data['theta']
            original_size = data['original_size']
            c0, c1 = data['pad_offset']
            
            print(f"Loaded sinogram shape: {mySino.shape}")
            
            start_time = time.perf_counter()
            
            print('Filtering...')
            filtSino = projFilter_gpu(mySino)
            
            print('Backprojection...')
            recon = backProj_gpu(filtSino, theta)
            
            end_time = time.perf_counter()
            print(f"Reconstruction time: {end_time - start_time:.6f} seconds")
            
            # Post-process and crop back to original size
            recon2 = np.round((recon - np.min(recon)) / np.ptp(recon) * 255)
            reconImg = Image.fromarray(recon2.astype('uint8'))
            reconImg = reconImg.crop((c0, c1, c0 + original_size[0], c1 + original_size[1]))
            
            # Save result
            save_comparison = False  # Set to True if you want to save side-by-side comparison
            if save_comparison:
                input_filename = Path(sino_path).stem.replace('_sinogram', '')
                output_filename = f"{input_filename}_reconstructed.png"
                output_path = os.path.join(output_folder, output_filename)
                reconImg.save(output_path)
            
                print(f"Saved reconstructed image: {output_path}")
                
        except Exception as e:
            print(f"Error processing {sino_path}: {str(e)}")
            continue
    
    print(f"\n--- Reconstruction complete! ---")
    print(f"Results saved in: {output_folder}")