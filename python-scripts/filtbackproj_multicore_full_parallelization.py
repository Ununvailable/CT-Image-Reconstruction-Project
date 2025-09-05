import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.fftpack import fft, fftshift, ifft
import time

from joblib import Parallel, delayed
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


def arange2(start, stop=None, step=1):
    if stop is None:
        a = np.arange(start)
    else:
        a = np.arange(start, stop, step)
        if a[-1] > stop - step:
            a = np.delete(a, -1)
    return a


def _filter_single_angle(proj_col, filt):
    projfft = fft(proj_col)
    filtProj = projfft * filt
    return np.real(ifft(filtProj))


def projFilter(sino, n_jobs):
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

    # Parallel filtering
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_filter_single_angle)(sino[:, i], filt) for i in range(numAngles)
    )

    filtSino = np.stack(results, axis=1)
    return filtSino


def _backproj_single_angle(sinogram_col, theta_rad, X, Y, imageLen):
    Xrot = X * np.sin(theta_rad) - Y * np.cos(theta_rad)
    XrotCor = np.round(Xrot + imageLen / 2).astype(int)
    projMatrix = np.zeros((imageLen, imageLen))
    mx, my = np.where((XrotCor >= 0) & (XrotCor <= (imageLen - 1)))
    projMatrix[mx, my] = sinogram_col[XrotCor[mx, my]]

    # print(f"Backprojection theta: {theta_rad}")
    return projMatrix


def backproject(sinogram, theta, n_jobs=None):
    imageLen = sinogram.shape[0]
    reconMatrix = np.zeros((imageLen, imageLen))

    x = np.arange(imageLen) - imageLen / 2
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    theta_rad = np.deg2rad(theta)
    numAngles = len(theta_rad)

    # Parallel back-projection
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_backproj_single_angle)(sinogram[:, n], theta_rad[n], X, Y, imageLen)
        for n in range(numAngles)
    )

    for mat in results:
        reconMatrix += mat

    backprojArray = np.flipud(reconMatrix)
    return backprojArray


if __name__ == '__main__':
    start_time = time.perf_counter()
    num_cores = multiprocessing.cpu_count()
    # num_cores = 2
    print(f"Detected CPU cores: {num_cores}")

    # myImg = dummyImg(400, 400)
    myImg = Image.open('data/phantoms/004085_01_02_107.png').convert('L')

    myImgPad, c0, c1 = padImage(myImg)
    dTheta = 0.1
    theta = np.arange(0, 361, dTheta)

    print('Getting projections (parallel)...')
    mySino = getProj(myImgPad, theta, n_jobs=num_cores)

    print('Filtering (parallel)...')
    filtSino = projFilter(mySino, n_jobs=num_cores)

    print('Performing backprojection (parallel)...')
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
