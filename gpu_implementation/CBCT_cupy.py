import numpy as np
import os
import time
import cupy as cp
from PIL import Image
from tqdm import tqdm
import logging
import pickle
import glob
from dataclasses import dataclass, asdict, field
from typing import Tuple, Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def discover_projection_files(folder: str, allowed_exts=(".tiff", ".tif", ".jpg", ".jpeg", ".png")) -> List[str]:
    pattern = os.path.join(folder, "*")
    files = [f for f in sorted(glob.glob(pattern)) if os.path.splitext(f)[1].lower() in allowed_exts]
    return files

@dataclass
class CBCTConfig:
    # Voxel/volume setup
    num_voxels: Tuple[int, int, int] = (256, 256, 256)
    volume_size_mm: Tuple[float, float, float] = (0.8, 50.0, 78.0)  # (sx, sy, sz) from main.py
    # Pixel/detector setup
    detector_pixels: Tuple[int, int] = (625, 512)
    detector_size_mm: Tuple[float, float] = (430.0, 350.0)  # (su, sv) from main.py
    detector_offset: Tuple[float, float] = (0.0, 0.0)
    # Acquisition geometry
    num_projections: int = 360
    angle_step: float = 1.0
    source_origin_dist: float = 212.515  # mm (DSO)
    source_detector_dist: float = 1304.5  # mm (DSD)
    # Preprocessing
    cosine_weighting: bool = True
    dark_current: float = 0.0
    bad_pixel_threshold: Optional[int] = None
    apply_log_correction: bool = False
    apply_bad_pixel_correction: bool = False
    apply_noise_reduction: bool = False
    apply_truncation_correction: bool = False
    truncation_width: int = 0
    # Filtering
    filter_type: str = 'none'  # 'none', 'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'
    # Saving
    save_intermediate: bool = True
    intermediate_path: str = "intermediate_results"
    output_path: str = "cbct_results"
    # Other
    max_gpu_memory_fraction: float = 0.8

    def derived(self) -> Dict[str, Any]:
        # Compute derived parameters similar to main.py, for snapshotting
        nx, ny, nz = self.num_voxels
        sx, sy, sz = self.volume_size_mm
        nu, nv = self.detector_pixels
        su, sv = self.detector_size_mm
        dx, dy, dz = sx / nx, sy / ny, sz / nz
        du, dv = su / nu, sv / nv
        return {
            'param_dx': dx, 'param_dy': dy, 'param_dz': dz,
            'param_du': du, 'param_dv': dv,
            'param_us': np.arange((-nu/2 + 0.5), (nu/2), 1) * du + self.detector_offset[0],
            'param_vs': np.arange((-nv/2 + 0.5), (nv/2), 1) * dv + self.detector_offset[1],
            'param_xs': np.arange((-nx/2 + 0.5), (nx/2), 1) * dx,
            'param_ys': np.arange((-ny/2 + 0.5), (ny/2), 1) * dy,
            'param_zs': np.arange((-nz/2 + 0.5), (nz/2), 1) * dz,
        }

class CBCTDataLoader:
    def __init__(self, config: CBCTConfig):
        self.config = config

    def load_projection_stack(self, folder: str) -> cp.ndarray:
        files = discover_projection_files(folder)
        if len(files) < self.config.num_projections:
            raise RuntimeError(f"Not enough projection files found in {folder} ({len(files)} found, "
                               f"{self.config.num_projections} required)")
        files = files[:self.config.num_projections]
        # Use PIL to load images, convert to float32, stack to (num_projections, nv, nu)
        logger.info(f"Loading {len(files)} projection images...")
        imgs = []
        for f in tqdm(files, desc="Loading projections"):
            im = np.array(Image.open(f).convert("F"))
            if im.shape != self.config.detector_pixels[::-1]:  # PIL loads as (height, width)
                raise ValueError(f"Projection image {f} has incorrect shape {im.shape}, expected {self.config.detector_pixels[::-1]}")
            imgs.append(im)
        arr = np.stack(imgs, axis=0)  # (nProj, nv, nu)
        arr = cp.array(arr, dtype=cp.float32)
        return arr  # (nProj, nv, nu)

class CBCTPreprocessor:
    def __init__(self, config: CBCTConfig):
        self.config = config
        self.derived = config.derived()

    def preprocess_stack(self, stack: cp.ndarray) -> cp.ndarray:
        # stack is (nProj, nv, nu)
        nProj, nv, nu = stack.shape
        proj_pre = cp.empty_like(stack)
        CW = 1.0
        if self.config.cosine_weighting:
            # Compute cosine weighting matrix (nv, nu)
            uu, vv = cp.meshgrid(self.derived['param_us'], self.derived['param_vs'])
            CW = self.config.source_detector_dist / cp.sqrt((uu ** 2 + vv ** 2) + self.config.source_detector_dist ** 2)
        for i in tqdm(range(nProj), desc="Preprocessing projections"):
            proj = stack[i]
            if self.config.cosine_weighting:
                proj = proj * CW
            # Add more steps as needed, e.g., log correction, noise reduction, etc.
            proj_pre[i] = proj
        return proj_pre

def ramp_flat(m):
    m = int(m / 2)
    a = np.zeros(m, order='C')
    for n in range(0, m):
        if n % 2 == 0:
            a[n] = 0
        else:
            a[n] = -1 / ((np.pi * n) ** 2)
    a[0] = 1 / 4
    to_add = -1 / ((np.pi * m) ** 2)
    b = np.delete(a, 0)[::-1]
    c = np.concatenate((b, a), axis=None)
    c = np.insert(c, 0, to_add)
    c = np.concatenate(([0], b, c[m:]), axis=None)
    return c

def get_filter_kernel(filter_type: str, order: int, d: float):
    kernel = ramp_flat(order)
    f_kernel = np.abs(np.fft.fft(kernel)) * 2
    filt = f_kernel[0:int(order / 2) + 1]
    filtnone = np.ones((filt.shape))
    ww = 2 * np.pi / order * np.arange(len(filt))
    if filter_type == 'none':
        filt = filtnone
    elif filter_type == 'ram-lak':
        pass
    elif filter_type == 'shepp-logan':
        filt[1:] = filt[1:] * (np.sin(ww[1:] / (2 * d)) / (ww[1:] / (2 * d)))
    elif filter_type == 'cosine':
        filt[1:] = filt[1:] * (np.cos(ww[1:] / (2 * d)))
    elif filter_type == 'hamming':
        filt[1:] = filt[1:] * (0.54 + 0.46 * np.cos(ww[1:] / d))
    elif filter_type == 'hann':
        filt[1:] = filt[1:] * (1 + np.cos(ww[1:] / d)) / 2
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    filt[ww > np.pi * d] = 0
    filt = np.concatenate((filt, filt[-2:0:-1]))
    return cp.array(filt, dtype=cp.float32)

class CBCTRampFilter:
    def __init__(self, config: CBCTConfig):
        self.config = config

    def filter_stack(self, stack: cp.ndarray) -> cp.ndarray:
        # stack is (nProj, nv, nu)
        nProj, nv, nu = stack.shape
        filt_len = 2 ** int(np.ceil(np.log2(2 * nu)))
        filter_kernel = get_filter_kernel(self.config.filter_type, filt_len, 1)
        start = int(filt_len/2 - nu/2)
        end = int(filt_len/2 + nu/2)
        filtered = cp.zeros_like(stack)
        for i in tqdm(range(nProj), desc="Filtering projections"):
            proj = stack[i]
            fproj = cp.zeros((nv, filt_len), dtype=cp.float32)
            fproj[:, start:end] = proj
            fproj = cp.fft.fft(fproj, axis=1)
            fproj = fproj * filter_kernel[None, :]
            fproj = cp.real(cp.fft.ifft(fproj, axis=1))
            filtered[i] = fproj[:, start:end]
        return filtered

class CBCTBackprojector:
    def __init__(self, config: CBCTConfig):
        self.config = config
        self.derived = config.derived()

    def backproject(self, filtered: cp.ndarray) -> np.ndarray:
        # Implementation closely follows main.py's backprojection
        nProj, nv, nu = filtered.shape
        nx, ny, nz = self.config.num_voxels
        param_us = cp.array(self.derived['param_us'])
        param_vs = cp.array(self.derived['param_vs'])
        param_xs = cp.array(self.derived['param_xs'])
        param_ys = cp.array(self.derived['param_ys'])
        param_zs = cp.array(self.derived['param_zs'])
        param_du = self.derived['param_du']
        param_dv = self.derived['param_dv']
        param_DSD = self.config.source_detector_dist
        param_DSO = self.config.source_origin_dist

        angle_step = self.config.angle_step
        nProj = filtered.shape[0]
        angle_rads = cp.array([np.pi * (i * angle_step / 180 - 0.5) for i in range(nProj)])

        r_cos_ls = cp.cos(angle_rads)[:, cp.newaxis, cp.newaxis]
        r_sin_ls = cp.sin(angle_rads)[:, cp.newaxis, cp.newaxis]
        xx, yy = cp.meshgrid(param_xs, param_ys)
        xx = cp.repeat(xx[cp.newaxis, ...], nProj, axis=0)
        yy = cp.repeat(yy[cp.newaxis, ...], nProj, axis=0)
        rx_ls = xx * r_cos_ls + yy * r_sin_ls
        ry_ls = -xx * r_sin_ls + yy * r_cos_ls
        pu_ls = (rx_ls * param_DSD / (ry_ls + param_DSO) + param_us[0]) / (-param_du)
        Ratio_ls = param_DSO ** 2 / (param_DSO + ry_ls) ** 2

        pu_ls_0 = cp.floor(pu_ls)
        pu_ls_1 = pu_ls_0 + 1
        pu_ls_0 = cp.clip(pu_ls_0, 0, nu - 1)
        pu_ls_1 = cp.clip(pu_ls_1, 0, nu - 1)
        pu_ls_0_int = pu_ls_0.astype(cp.int32)
        pu_ls_1_int = pu_ls_1.astype(cp.int32)
        x1_x = pu_ls_1 - pu_ls
        x_x0 = pu_ls - pu_ls_0
        cond_0 = (pu_ls <= 0) + (pu_ls >= nu)
        vols = []

        for i in tqdm(range(nz), desc="Backprojecting"):
            var1 = param_DSD / (ry_ls + param_DSO)
            var2 = param_vs[0]
            pv_ls = ((param_zs[i] * var1 - var2) / param_dv)
            pv_ls_0 = cp.floor(pv_ls)
            pv_ls_1 = pv_ls_0 + 1
            pv_ls_0 = cp.clip(pv_ls_0, 0, nv - 1)
            pv_ls_1 = cp.clip(pv_ls_1, 0, nv - 1)
            pv_ls_0_int = pv_ls_0.astype(cp.int32)
            pv_ls_1_int = pv_ls_1.astype(cp.int32)
            y1_y = pv_ls_1 - pv_ls
            y_y0 = pv_ls - pv_ls_0
            wa = x1_x * y1_y
            wb = x1_x * y_y0
            wc = x_x0 * y1_y
            wd = x_x0 * y_y0
            cond_1 = (pv_ls <= 0) + (pv_ls >= nv)
            cond = (cond_0 + cond_1) == False

            Ia = cp.array([filtered[t][pv_ls_0_int[t], pu_ls_0_int[t]] for t in range(nProj)])
            Ib = cp.array([filtered[t][pv_ls_1_int[t], pu_ls_0_int[t]] for t in range(nProj)])
            Ic = cp.array([filtered[t][pv_ls_0_int[t], pu_ls_1_int[t]] for t in range(nProj)])
            Id = cp.array([filtered[t][pv_ls_1_int[t], pu_ls_1_int[t]] for t in range(nProj)])

            k = cp.sum(Ratio_ls * cond * (Ia * wa + Ib * wb + Ic * wc + Id * wd), axis=0)
            vols.append(k)
        vols = cp.array(vols)
        return cp.asnumpy(vols)

def save_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def save_config_snapshot(config: CBCTConfig, runtime_info: Dict[str, Any], path: str):
    snapshot = asdict(config)
    snapshot.update(runtime_info)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(snapshot, f)

def main():
    config = CBCTConfig()
    os.makedirs(config.output_path, exist_ok=True)
    os.makedirs(config.intermediate_path, exist_ok=True)
    data_loader = CBCTDataLoader(config)
    preprocessor = CBCTPreprocessor(config)
    ramp_filter = CBCTRampFilter(config)
    backprojector = CBCTBackprojector(config)
    t0 = time.time()
    projections = data_loader.load_projection_stack("projection_data")
    if config.save_intermediate:
        save_pickle(cp.asnumpy(projections), os.path.join(config.intermediate_path, "projections_raw.pickle"))

    preprocessed = preprocessor.preprocess_stack(projections)
    if config.save_intermediate:
        save_pickle(cp.asnumpy(preprocessed), os.path.join(config.intermediate_path, "projections_preprocessed.pickle"))

    filtered = ramp_filter.filter_stack(preprocessed)
    if config.save_intermediate:
        save_pickle(cp.asnumpy(filtered), os.path.join(config.intermediate_path, "projections_filtered.pickle"))

    reconstructed = backprojector.backproject(filtered)
    save_pickle(reconstructed, os.path.join(config.output_path, "volume.pickle"))

    t1 = time.time()
    runtime_info = {
        'run_time_seconds': t1 - t0
    }
    save_config_snapshot(config, runtime_info, os.path.join(config.output_path, "config_snapshot.pickle"))
    logger.info(f"Reconstruction complete, total time: {t1 - t0:.2f} seconds")
    logger.info(f"Volume saved at {os.path.join(config.output_path, 'volume.pickle')}")
    logger.info(f"Config snapshot saved at {os.path.join(config.output_path, 'config_snapshot.pickle')}")

if __name__ == '__main__':
    main()