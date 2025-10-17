import math
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
import datetime
import tqdm
import psutil
import cv2
import os
import pickle

num_cpus = psutil.cpu_count(logical=False)
starttime = datetime.datetime.now()

########    Voxel Setup
param_nx = 256  # number of voxels.  What's implied is nx = ny = nz
param_ny = 256
param_nz = 256

# The Al extrusion is square in x-y with side = 20 mm and length = 86 mm
# param_sx = 20   # mm (real size of the object)
# param_sy = 20
# param_sz = 86

param_sx = 0.8  # mm (set size of the object)
param_sy = 50
param_sz = 78

# param_sx = 80   # mm (set size of the object)
# param_sy = 80
# param_sz = 80
########    Voxel Setup end


########    Pixel Setup
# param_nu = 3052  ## real number of pixels along u
# param_nv = 2500  ## real number of pixels along v
# param_nu = 128     ## squeezed-down number of pixels along u
# param_nv = 105     ## squeezed-down number of pixels along v
param_nu = 625
param_nv = 512

# Active area's dimensions
param_su = 430  ##size of detector in u (in units of mm)
param_sv = 350  ##size of detector in v (in units of mm)

param_off_u = 0  ## offset in u
param_off_v = 0  ## offset in v

########    Pixel Setup End

# Define the total number of projections as nProj
#nProj = 36
#angle_step = 10  # in degrees
nProj = 360
angle_step = 1  # in degrees
#############   Distances   ################################################
param_DSO = 212.515  ###Distance X-ray source to Origin   (in units of mm)
param_DSD = 1304.5  ###Distance X-ray source to Detector (in units of mm)
# param_DSO = 150
# param_DSD = 600
############################################################################

############################ DERIVED detector geometry info @@@@@@@@@@@@@@@@@@@@@

#########   pixel size
param_du = param_su / param_nu  ##size of each pixel along u in mm
param_dv = param_sv / param_nv  ##size of each pixel along v in mm

#########   pixel coordinates (with offsets)
param_us = np.arange((-param_nu / 2 + 0.5), (param_nu / 2), 1) * param_du + param_off_u
param_vs = np.arange((-param_nv / 2 + 0.5), (param_nv / 2), 1) * param_dv + param_off_v

############################ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
filter_used = 'none'
# filter_used = 'ram-lak' 2477
# filter_used = 'shepp-logan' 2338
# filter_used = 'cosine' 2328
# filter_used = 'hamming' 2302
# filter_used = 'hann'   #2455
#########filter options (pick one!):
'''
'none'
'ram-lak'
'shepp-logan'
'cosine'
'hamming'
'hann'

'''
#######


########################### DERIVED quantities of voxels@@@@@@@@@@@@@@@@@@@@@@@@@
#########   Voxel size as belows:
param_dx = param_sx / param_nx
param_dy = param_sy / param_ny
param_dz = param_sz / param_nz

########### Cordinates for the voxels
param_xs = np.arange((-param_nx / 2 + 0.5), (param_nx / 2), 1) * param_dx
param_ys = np.arange((-param_ny / 2 + 0.5), (param_ny / 2), 1) * param_dy
param_zs = np.arange((-param_nz / 2 + 0.5), (param_nz / 2), 1) * param_dz

us = np.arange((-param_nu / 2 + 0.5), (param_nu / 2), 1) * param_du + param_off_u

vs = np.arange((-param_nv / 2 + 0.5), (param_nv / 2), 1) * param_dv + param_off_v
uu, vv = np.meshgrid(us, vs)

# Cosine Weighting CW as below
CW = param_DSD / np.sqrt((np.array(uu * uu + vv * vv) + param_DSD * param_DSD))
filt_len = 2 ** math.ceil(math.log2(2 * param_nu))


def ramp_flat(m):
    m = int(m / 2)
    a = np.zeros(m, order='C')  # declare a zero array first
    for n in np.arange(0, m, 1):
        if n % 2 == 0:  # populate the array's elements in accordance with n (even or odd)
            a[n] = 0
        else:
            a[n] = -1 / (((math.pi) * n) ** 2)
    a[0] = 1 / 4

    to_add = -1 / -1 / (((math.pi) * m) ** 2)
    b = np.delete(a, 0)  # get rid of a's first element and call it b
    b = b[::-1]  # reverse b
    c = np.concatenate((b, a), axis=None)  # put b and a together
    c = np.insert(c, 0, to_add)  # place to_add at the very front of the 1D array, lacked one element
    d = c[::-1]
    # c = np.concatenate((c[m:], [0], d[:(m-1)]), axis=None)
    c = np.concatenate(([0], b, c[m:]), axis=None)
    return c


def Filter(filter, kernel, order, d):
    f_kernel = np.abs(np.fft.fft(kernel)) * 2

    filt = f_kernel[0:int(order / 2) + 1]
    filtnone = np.ones((filt.shape))

    ww = 2 * (np.pi) / order * np.array(range(len(filt)))
    if filter == 'none':
        filt = filtnone
    elif filter == 'ram-lak':
        filt = filt
    elif filter == 'shepp-logan':
        filt[1:] = filt[1:] * (np.sin(ww[1:] / (2 * d)) / (ww[1:] / (2 * d)))

    elif filter == 'cosine':
        filt[1:] = filt[1:] * (np.cos(ww[1:] / (2 * d)))

    elif filter == 'hamming':
        filt[1:] = filt[1:] * (0.54 + 0.46 * np.cos(ww[1:] / (d)))

    elif filter == 'hann':
        filt[1:] = filt[1:] * (1 + np.cos(ww[1:] / (d))) / 2

    else:
        print('invalid filter selected!!!')

    filt[ww > (np.pi) * d] = 0
    filt = np.concatenate((filt, filt[-2:0:-1]))
    return filt


# Define the composite projection called proj, a 3D array
# proj = np.zeros((nProj, param_nv, param_nu), order='C')
fproj_start = int(filt_len / 2 - param_nu / 2)
fproj_end = int(filt_len / 2 + param_nu / 2)
proj_divided_param = 2 / param_du * (2 * np.pi / nProj) / 2 * (param_DSD / param_DSO)


def load_img( path_ls):
    proj = cv2.imread(path_ls, 0) * CW
    fproj = np.zeros((param_nv, filt_len), order='C')

    # replace the middle block of O's with CWed proj
    fproj[:, fproj_start:fproj_end] = proj

    # FFT above and still call it fproj
    fproj = np.fft.fft(fproj)

    # multiplication of 2 ffts below.  Filter was FFTed in above function (check it out)
    fproj = fproj * Filter(filter_used, ramp_flat(filt_len), filt_len, 1)

    # Do IFFT for above
    fproj = np.real(np.fft.ifft(fproj))

    # Below, carve out the middle block as the results before doing back projection
    proj = fproj[:, fproj_start:fproj_end]
    proj = proj / proj_divided_param

    return proj


path_ls = ['data/20200225_AXI_final_code/slices' + str(i) + '.jpg' for i in range(nProj)]
print('AAAAAAAAAAAAAAAAAAAAA')
if __name__ == '__main__':
    start = time.time()
    with Pool(num_cpus) as pool:
        proj = list(pool.map(load_img, path_ls))
    proj = cp.array(proj)
    print('Load Image cost: ',time.time() - start)

    param_xs = cp.array(param_xs)
    param_ys = cp.array(param_ys)

    nProj = len(proj)
    angle_rads = cp.array([np.pi * (i * angle_step / 180 - 0.5) for i in range(nProj)])  # nProj

    r_cos_ls = cp.cos(angle_rads)[:, cp.newaxis, cp.newaxis]  # (nProj,1,1)
    r_sin_ls = cp.sin(angle_rads)[:, cp.newaxis, cp.newaxis]  # (nProj,1,1)

    xx, yy = cp.meshgrid(param_xs, param_ys)  # (128,128), (128,128)
    xx = cp.repeat(xx[cp.newaxis, ...], nProj, axis=0)  # (nProj,128,128)
    yy = cp.repeat(yy[cp.newaxis, ...], nProj, axis=0)  # (nProj,128,128)

    rx_ls = xx * r_cos_ls + yy * r_sin_ls  # (nProj,128,128)
    ry_ls = -xx * r_sin_ls + yy * r_cos_ls  # (nProj,128,128)

    pu_ls = (rx_ls * param_DSD / (ry_ls + param_DSO) + param_us[0]) / (-param_du)  # (nProj,128,128)
    Ratio_ls = param_DSO ** 2 / (param_DSO + ry_ls) ** 2  # (nProj,128,128)

    var1 = param_DSD / (ry_ls + param_DSO)
    var2 = param_vs[0]

    pu_ls_0 = cp.floor(pu_ls)
    pu_ls_1 = pu_ls_0 + 1

    pu_ls_0 = cp.clip(pu_ls_0, 0, param_nu - 1)
    pu_ls_1 = cp.clip(pu_ls_1, 0, param_nu - 1)

    pu_ls_0_int = pu_ls_0.astype(cp.int)
    pu_ls_1_int = pu_ls_1.astype(cp.int)

    x1_x = pu_ls_1 - pu_ls
    x_x0 = pu_ls - pu_ls_0

    cond_0 = (pu_ls <= 0) + (pu_ls >= param_nu)

    vols = []

    start = time.time()
    for i in tqdm.tqdm(range(param_nz)):
        pv_ls = ((param_zs[i] * var1 - var2) / param_dv)

        pv_ls_0 = cp.floor(pv_ls)
        pv_ls_1 = pv_ls_0 + 1

        pv_ls_0 = cp.clip(pv_ls_0, 0, param_nv - 1)
        pv_ls_1 = cp.clip(pv_ls_1, 0, param_nv - 1)

        pv_ls_0_int = pv_ls_0.astype(cp.int)
        pv_ls_1_int = pv_ls_1.astype(cp.int)

        y1_y = pv_ls_1_int - pv_ls
        y_y0 = pv_ls - pv_ls_0_int

        wa = x1_x * y1_y
        wb = x1_x * y_y0
        wc = x_x0 * y1_y
        wd = x_x0 * y_y0

        cond_1 = (pv_ls <= 0) + (pv_ls >= param_nv)
        cond = (cond_0 + cond_1) == False

        Ia = cp.array([proj[t][pv_ls_0_int[t], pu_ls_0_int[t]] for t in range(nProj)])
        Ib = cp.array([proj[t][pv_ls_1_int[t], pu_ls_0_int[t]] for t in range(nProj)])
        Ic = cp.array([proj[t][pv_ls_0_int[t], pu_ls_1_int[t]] for t in range(nProj)])
        Id = cp.array([proj[t][pv_ls_1_int[t], pu_ls_1_int[t]] for t in range(nProj)])

        k = cp.sum(Ratio_ls * cond * (Ia * wa + Ib * wb + Ic * wc + Id * wd), axis=0)
        vols.append(k)

    vols = cp.array(vols)
    vols = cp.asnumpy(vols)
    print('Time cost: ', time.time() - start)
    
    with open('data/20200225_AXI_final_code/results/vols.pickle', 'wb') as f:
        pickle.dump(vols, f)
    
    




