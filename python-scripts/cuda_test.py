import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.fftpack import fft, fftshift, ifft
import time

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
    # cpu_cores = 1
    print(f"Detected CPU cores: {cpu_cores}")

    # myImg = dummyImg(400, 400)
    myImg = Image.open('data/phantoms/004085_01_02_107.png').convert('L')

    myImgPad, c0, c1 = padImage(myImg)
    dTheta = 0.1
    theta = np.arange(0, 361, dTheta)

    print('Getting projections (sequential)...')
    mySino = getProj_cpu(myImgPad, theta, n_jobs=cpu_cores)  # CPU
    # mySino = getProj(myImgPad, theta, True, n_jobs=cpu_cores)  # GPU
    
    start_time = time.perf_counter()
    
    print('Filtering (parallel)...')
    # filtSino = projFilter(mySino, n_jobs=num_cores)  # CPU
    filtSino = projFilter_gpu(mySino)  # GPU

    print('Performing backprojection (parallel)...')
    # recon = backProj(filtSino, theta, n_jobs=num_cores)  # CPU
    recon = backProj_gpu(filtSino, theta)  # GPU

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

    recon2 = np.round((recon - np.min(recon)) / np.ptp(recon) * 255)
    reconImg = Image.fromarray(recon2.astype('uint8'))
    n0, n1 = myImg.size
    reconImg = reconImg.crop((c0, c1, c0 + n0, c1 + n1))

    fig3, (ax1, ax2, ax3, ax4, ax5)= plt.subplots(1, 5, figsize=(12, 4))
    ax1.imshow(myImg, cmap='gray')
    ax1.set_title('Original Image')
    ax2.imshow(reconImg, cmap='gray')
    ax2.set_title('Filtered Backprojected Image')
    ax3.imshow(ImageChops.difference(myImg, reconImg), cmap='gray')
    ax3.set_title('Error')
    ax4.imshow(mySino, cmap='gray')
    ax4.set_title('Captured Sinogram')
    ax5.imshow(filtSino, cmap='gray')
    ax5.set_title('Filtered Sinogram')
    plt.show()
