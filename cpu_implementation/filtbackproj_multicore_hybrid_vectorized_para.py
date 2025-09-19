import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.fftpack import fft, fftshift, ifft
from multiprocessing import Pool
import multiprocessing
import time


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


def _single_proj(args):
    img_arr, angle_deg = args
    img_rot = Image.fromarray(img_arr).rotate(90 - angle_deg, resample=Image.BICUBIC)
    return np.sum(np.array(img_rot), axis=0)


def getProj(img, theta, n_jobs=None, show_progress=False):
    """
    Parallel Radon projection using multiprocessing Pool.
    """
    numAngles = len(theta)
    img_arr = np.array(img)
    
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    
    # Prepare arguments for Pool.map
    args = [(img_arr, theta[n]) for n in range(numAngles)]
    
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_single_proj, args)
    
    sinogram = np.stack(results, axis=1)
    return sinogram


def arange2(start, stop=None, step=1):
    if stop is None:
        a = np.arange(start)
    else:
        a = np.arange(start, stop, step)
        if a[-1] > stop - step:
            a = np.delete(a, -1)
    return a


def projFilter(sino, n_jobs=None):
    """
    Vectorized filtering - much faster than parallel for this operation.
    """
    a = 0.5
    projLen, numAngles = sino.shape
    step = 2 * np.pi / projLen
    w = arange2(-np.pi, np.pi, step)
    if len(w) < projLen:
        w = np.concatenate([w, [w[-1] + step]])

    rn1 = abs(2 / a * np.sin(a * w / 2))
    rn2 = np.sin(a * w / 2) / (a * w / 2)
    r = rn1 * (rn2) ** 2
    filt = fftshift(r)

    # Vectorized: single FFT for all columns
    sino_fft = fft(sino, axis=0)
    filtered_fft = sino_fft * filt[:, np.newaxis]
    return np.real(ifft(filtered_fft, axis=0))


def _backproj_angles_chunk(args):
    sinogram_chunk, theta_chunk, X, Y, imageLen = args
    reconMatrix = np.zeros((imageLen, imageLen))
    
    for i, angle in enumerate(theta_chunk):
        theta_rad = np.deg2rad(angle)
        Xrot = X * np.sin(theta_rad) - Y * np.cos(theta_rad)
        XrotCor = np.round(Xrot + imageLen / 2).astype(int)
        
        mx, my = np.where((XrotCor >= 0) & (XrotCor <= (imageLen - 1)))
        projMatrix = np.zeros((imageLen, imageLen))
        projMatrix[mx, my] = sinogram_chunk[XrotCor[mx, my], i]
        reconMatrix += projMatrix
    
    return reconMatrix


def backproject(sinogram, theta, n_jobs=None):
    imageLen = sinogram.shape[0]
    numAngles = len(theta)
    
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    
    x = np.arange(imageLen) - imageLen / 2
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    
    # Only use multiprocessing for large angle counts
    if numAngles < 100:
        # Single-threaded for small datasets
        reconMatrix = np.zeros((imageLen, imageLen))
        for n in range(numAngles):
            theta_rad = np.deg2rad(theta[n])
            Xrot = X * np.sin(theta_rad) - Y * np.cos(theta_rad)
            XrotCor = np.round(Xrot + imageLen / 2).astype(int)
            
            mx, my = np.where((XrotCor >= 0) & (XrotCor <= (imageLen - 1)))
            projMatrix = np.zeros((imageLen, imageLen))
            projMatrix[mx, my] = sinogram[XrotCor[mx, my], n]
            reconMatrix += projMatrix
        
        return np.flipud(reconMatrix)
    
    # Parallel processing for large datasets
    # Use fewer, larger chunks to reduce overhead
    # chunk_size = max(50, numAngles // (n_jobs // 2))  # Bigger chunks
    chunk_size = 100
    chunks = []
    
    for i in range(0, numAngles, chunk_size):
        end_idx = min(i + chunk_size, numAngles)
        sino_chunk = sinogram[:, i:end_idx]
        theta_chunk = theta[i:end_idx]
        chunks.append((sino_chunk, theta_chunk, X, Y, imageLen))
    
    with Pool(processes=min(n_jobs, len(chunks))) as pool:
        results = pool.map(_backproj_angles_chunk, chunks)
    
    reconMatrix = np.sum(results, axis=0)
    return np.flipud(reconMatrix)


if __name__ == '__main__':
    start_time = time.perf_counter()
    num_cores = multiprocessing.cpu_count()
    print(f"Detected CPU cores: {num_cores}")

    # myImg = dummyImg(400, 400)
    myImg = Image.open('data/phantoms/004085_01_02_107.png').convert('L')

    myImgPad, c0, c1 = padImage(myImg)
    dTheta = 0.1
    theta = np.arange(0, 361, dTheta)

    print('Getting projections (parallel)...')
    mySino = getProj(myImgPad, theta, n_jobs=num_cores)

    print('Filtering (vectorized)...')
    filtSino = projFilter(mySino, n_jobs=num_cores)

    print('Performing backprojection (selective parallel)...')
    recon = backproject(filtSino, theta, n_jobs=num_cores)

    recon2 = np.round((recon - np.min(recon)) / np.ptp(recon) * 255)
    reconImg = Image.fromarray(recon2.astype('uint8'))
    n0, n1 = myImg.size
    reconImg = reconImg.crop((c0, c1, c0 + n0, c1 + n1))

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

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