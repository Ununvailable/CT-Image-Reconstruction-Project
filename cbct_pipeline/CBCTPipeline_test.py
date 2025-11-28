#!/usr/bin/env python3
"""
CBCT pipeline with CuPy backprojector using manual array bilinear sampling.

This script:
- Preserves metadata-driven config, data loading, preprocessing, FFT ramp filtering.
- Uses explicit Geometry and Projector separation (ASTRA-like).
- Implements a per-projection backprojector that uses a GPU kernel which samples the
  projection from a linear device buffer (manual bilinear interpolation).
- Removes texture sampling entirely (no cudaArray / texref usage).
- Accumulates into the volume with atomicAdd per projection.

Notes:
- Detector offset is taken from metadata as millimetres (mm). If the metadata stores
  detector offsets in pixels, convert them to mm before running or set `detector_offset_in_pixels=True`.
- This version is intended to be easier to run across environments because it avoids texture APIs
  that vary between CUDA/CuPy versions.
- For large volumes or many projections, performance tuning and alternative strategies (per-block
  reductions, shared-memory accumulation, tiled projections) are still recommended.

Run:
    python cbct_pipeline_array_kernel.py --input <projections_folder> --metadata metadata.json
"""

import os
import sys
import time
import json
import logging
from dataclasses import asdict, dataclass
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import cupy as cp
from PIL import Image
from tqdm import tqdm
import tifffile

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cbct_array")

# ---------------------------
# Configuration & metadata
# ---------------------------

@dataclass
class CBCTConfig:
    # Volume (nx, ny, nz) and physical sizes (mm)
    num_voxels: Tuple[int, int, int] = (256, 256, 256)
    voxel_size: Tuple[float, float, float] = (0.006134010138512811, 0.006134010138512811, 0.006134010138512811)  # mm, assume isotropic
    volume_size_mm: Tuple[float, float, float] = (17.543, 17.543, 17.543)

    # Detector: (nu, nv) = (width, height) in pixels, and physical size in mm (su, sv)
    detector_pixels: Tuple[int, int] = (715, 715)
    detector_size_mm: Tuple[float, float] = (430.0, 350.0)
    detector_offset: Tuple[float, float] = (0.0, 0.0)  # in mm by default

    # If metadata gives offsets in pixels set this True (then convert to mm = pixels * du)
    detector_offset_in_pixels: bool = True

    # Acquisition geometry
    num_projections: int = 360
    angle_step: float = 0.225       # degrees
    start_angle: float = 270.0      # degrees
    source_origin_dist: float = 28.625365287711134
    source_detector_dist: float = 699.9996522369905

    # Projection dtype for raw files
    projection_dtype: str = "uint16"

    # Preprocessing flags
    cosine_weighting: bool = True
    dark_current: float = 0.0
    bad_pixel_threshold: Optional[float] = None
    apply_log_correction: bool = True
    apply_bad_pixel_correction: bool = False
    apply_noise_reduction: bool = False
    apply_truncation_correction: bool = False
    truncation_width: int = 0

    # Filtering
    filter_type: str = "ram-lak"  # choose 'none', 'ram-lak', 'shepp-logan', ...

    # IO
    input_path: str = "data/slices"
    intermediate_path: str = "data/intermediate"
    output_path: str = "data/results"
    save_intermediate: bool = True

    # Misc
    max_gpu_memory_fraction: float = 0.8

    # Internal cache
    _derived_cache: Optional[Dict[str, Any]] = None
    _metadata_loaded: bool = False

    @classmethod
    def create_with_metadata(cls, folder: str, metadata_filename: str = "metadata.json"):
        cfg = cls()
        cfg.input_path = f"{folder}/slices"
        cfg.intermediate_path = f"{folder}/intermediate"
        cfg.output_path = f"{folder}/results"
        meta_path = f"{cfg.input_path}/{metadata_filename}"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            cfg._apply_metadata(meta)
            cfg._metadata_loaded = True
            logger.info(f"Loaded metadata from {meta_path}")
            logger.info(f"Configuration: {cfg}")
        else:
            logger.warning(f"No metadata file at {meta_path}, using defaults")
        return cfg

    def _apply_metadata(self, meta: Dict[str, Any]):
        """Apply metadata fields to config"""
        if "num_projections" in meta:
            self.num_projections = int(meta["num_projections"])
        if "angle_step" in meta:
            self.angle_step = float(meta["angle_step"])
        if "start_angle" in meta:
            self.start_angle = float(meta["start_angle"])
        if "scan_angle" in meta:
            self.scan_angle = float(meta["scan_angle"])
        if "detector_pixels" in meta:
            self.detector_size = tuple(int(x) for x in meta["detector_pixels"])
        if "detector_size_mm" in meta:
            det_size = tuple(float(x) for x in meta["detector_size_mm"])
            self.pixel_size_u = det_size[0] / self.detector_size[1]
            self.pixel_size_v = det_size[1] / self.detector_size[0]
        if "pixel_size_mm" in meta:
            pixel_sizes = meta["pixel_size_mm"]
            self.pixel_size_u = float(pixel_sizes[0])
            self.pixel_size_v = float(pixel_sizes[1])
        if "detector_offset" in meta:
            offsets = tuple(float(x) for x in meta["detector_offset"])
            self.detector_offset_u = offsets[0]
            self.detector_offset_v = offsets[1]
        if "volume_voxels" in meta:
            self.volume_size = tuple(int(x) for x in meta["volume_voxels"])
        if "voxel_size_mm" in meta:
            vs = meta["voxel_size_mm"]
            if isinstance(vs, (list, tuple)):
                self.voxel_size = float(vs[0])
            else:
                self.voxel_size = float(vs)
        if "source_origin_dist" in meta:
            self.source_object_dist = float(meta["source_origin_dist"])
        if "source_detector_dist" in meta:
            self.source_detector_dist = float(meta["source_detector_dist"])
        if "astra_downsample_factor" in meta:
            self.astra_downsample_factor = int(meta["astra_downsample_factor"])

        # Preprocessing parameters
        if "dark_current" in meta:
            self.dark_current = float(meta["dark_current"])
        if "apply_log_correction" in meta:
            self.apply_log_correction = bool(meta["apply_log_correction"])
        if "apply_bad_pixel_correction" in meta:
            self.apply_bad_pixel_correction = bool(meta["apply_bad_pixel_correction"])
        if "apply_noise_reduction" in meta:
            self.apply_noise_reduction = bool(meta["apply_noise_reduction"])
        if "apply_truncation_correction" in meta:
            self.apply_truncation_correction = bool(meta["apply_truncation_correction"])
        
        # RAW file parameters
        if "resolution" in meta:
            res = meta["resolution"]
            if isinstance(res, str):
                # Parse "3072 × 3072" format                                    
                parts = res.replace('×', 'x').split('x')
                self.raw_resolution = (int(parts[0].strip()), int(parts[1].strip()))
            else:
                self.raw_resolution = tuple(int(x) for x in res)
        if "bit_depth" in meta:
            self.raw_bit_depth = str(meta["bit_depth"]).replace("_t", "")
        if "endianness" in meta:
            self.raw_endianness = str(meta["endianness"]).lower()
        if "header_size" in meta:
            hs = meta["header_size"]
            if isinstance(hs, str):
                self.raw_header_size = int(hs.split()[0])  # "96 bytes"
            else:
                self.raw_header_size = int(hs)
        if "projection_dtype" in meta:
            self.raw_bit_depth = str(meta["projection_dtype"])
    def derived(self) -> Dict[str, Any]:
        """Compute derived geometry variables and cache them."""
        if self._derived_cache is not None:
            return self._derived_cache

        nx, ny, nz = self.num_voxels
        sx, sy, sz = self.volume_size_mm
        nu, nv = self.detector_pixels
        su, sv = self.detector_size_mm

        dx = sx / nx
        dy = sy / ny
        dz = sz / nz
        du = su / nu
        dv = sv / nv

        # nx, ny, nz = self.num_voxels
        # sx, sy, sz = self.volume_size_mm
        # nu, nv = self.detector_pixels
        # su, sv = self.detector_size_mm
        # dx, dy, dz = self.voxel_size  # Override with explicit voxel size if provided
        # du, dv = self.detector_size_mm

        us = (np.arange(nu) - nu / 2.0 + 0.5) * du  # centered pixel centers (mm)
        vs = (np.arange(nv) - nv / 2.0 + 0.5) * dv

        xs = (np.arange(nx) - nx / 2.0 + 0.5) * dx
        ys = (np.arange(ny) - ny / 2.0 + 0.5) * dy
        zs = (np.arange(nz) - nz / 2.0 + 0.5) * dz

        self._derived_cache = {
            "dx": dx, "dy": dy, "dz": dz,
            "du": du, "dv": dv,
            "us": us, "vs": vs,
            "xs": xs, "ys": ys, "zs": zs,
        }
        return self._derived_cache


# ---------------------------
# Data loader
# ---------------------------

def discover_projection_files(folder: str, allowed_exts=(".tiff", ".tif", ".jpg", ".jpeg", ".png", ".raw")) -> List[str]:
    import glob
    pattern = os.path.join(folder, "*")
    files = [f for f in sorted(glob.glob(pattern)) if os.path.splitext(f)[1].lower() in allowed_exts]
    return files


class CBCTDataLoader:
    def __init__(self, config: CBCTConfig):
        self.config = config

    def _load_tiff(self, path: str) -> np.ndarray:
        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        return arr.astype(np.float32)

    def _load_image(self, path: str) -> np.ndarray:
        with Image.open(path) as img:
            return np.array(img.convert("F"), dtype=np.float32)

    def _load_raw(self, path: str, shape: Tuple[int, int]) -> np.ndarray:
        dtype = np.dtype(self.config.projection_dtype)
        expected = shape[0] * shape[1] * dtype.itemsize
        size = os.path.getsize(path)
        if size != expected:
            logger.warning(f"RAW file size mismatch for {path}: expected {expected}, got {size}")
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=dtype)
        try:
            data = data.reshape(shape)
        except Exception:
            data = data.reshape(shape[::-1]).T
        return data.astype(np.float32)

    def load_projection_stack(self, folder: Optional[str] = None, metadata_filename: str = "metadata.json") -> cp.ndarray:
        if folder is None:
            folder = self.config.input_path

        meta_path = f"{folder}/{metadata_filename}"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.config._apply_metadata(meta)
            logger.info(f"Applied metadata from {meta_path}")

        files = discover_projection_files(folder)
        if len(files) == 0:
            raise RuntimeError(f"No projection files found in {folder}")

        if len(files) < self.config.num_projections:
            logger.warning(f"Found {len(files)} projection files but config expects {self.config.num_projections}. Using available files.")
        files = files[: self.config.num_projections]

        expected_shape = (self.config.detector_pixels[1], self.config.detector_pixels[0])  # (height, width)
        imgs = []
        for path in tqdm(files, desc="Loading projections"):
            ext = os.path.splitext(path)[1].lower()
            if ext in (".tif", ".tiff"):
                img = self._load_tiff(path)
            elif ext in (".png", ".jpg", ".jpeg"):
                img = self._load_image(path)
            elif ext == ".raw":
                img = self._load_raw(path, expected_shape)
            else:
                raise RuntimeError(f"Unsupported projection format {ext}")

            if img.shape != expected_shape:
                img = np.array(Image.fromarray(img).resize((expected_shape[1], expected_shape[0]), Image.LANCZOS), dtype=np.float32)
            imgs.append(img)

        stack = np.stack(imgs, axis=0).astype(np.float32)   # (nProj, nv, nu)
        logger.info(f"Loaded projection stack shape {stack.shape}")
        return cp.asarray(stack)


# ---------------------------
# Preprocessing
# ---------------------------

class CBCTPreprocessor:
    def __init__(self, config: CBCTConfig):
        self.config = config
        self.derived = config.derived()

    def _dark_current(self, p: cp.ndarray) -> cp.ndarray:
        if self.config.dark_current != 0:
            p = p - float(self.config.dark_current)
            p = cp.maximum(p, 1.0)
        return p

    def _log(self, p: cp.ndarray, I0: Optional[float] = None) -> cp.ndarray:
        if I0 is None:
            I0 = float(cp.max(p))
        eps = 1e-6
        p = cp.maximum(p, eps)
        I0 = max(I0, eps)
        return cp.log(I0) - cp.log(p)

    def _median(self, p: cp.ndarray) -> cp.ndarray:
        try:
            from cupyx.scipy.ndimage import median_filter
            bad_mask = None
            if self.config.bad_pixel_threshold is not None:
                bad_mask = p > self.config.bad_pixel_threshold
            if bad_mask is None:
                return median_filter(p, size=3)
            else:
                filt = median_filter(p, size=3)
                return cp.where(bad_mask, filt, p)
        except Exception:
            logger.warning("cupyx median_filter unavailable; skipping bad pixel correction")
            return p

    def _gaussian(self, p: cp.ndarray) -> cp.ndarray:
        try:
            from cupyx.scipy.ndimage import gaussian_filter
            return gaussian_filter(p, sigma=1.0)
        except Exception:
            logger.warning("cupyx gaussian_filter unavailable; skipping noise reduction")
            return p

    def preprocess_stack(self, stack: cp.ndarray) -> cp.ndarray:
        nproj, nv, nu = stack.shape
        out = cp.empty_like(stack)
        # cosine weighting matrix (per projection independent)
        if self.config.cosine_weighting:
            us = cp.asarray(self.derived["us"], dtype=cp.float32)
            vs = cp.asarray(self.derived["vs"], dtype=cp.float32)
            UU, VV = cp.meshgrid(us, vs)  # shape (nv, nu) since vs is rows
            SD = float(self.config.source_detector_dist)
            CW = (SD / cp.sqrt(UU**2 + VV**2 + SD**2)).astype(cp.float32)
        else:
            CW = 1.0

        # Precompute I0 if doing log-correction: use max of first projection after dark correction
        I0 = None
        if self.config.apply_log_correction:
            first = stack[0].copy()
            if self.config.dark_current != 0:
                first = self._dark_current(first)
            I0 = float(cp.max(first))
            logger.info(f"Using I0={I0:.2f} for log-correction")

        for i in tqdm(range(nproj), desc="Preprocessing"):
            p = stack[i].copy()
            if self.config.dark_current != 0:
                p = self._dark_current(p)
            if self.config.apply_log_correction:
                p = self._log(p, I0)
            if self.config.apply_bad_pixel_correction:
                p = self._median(p)
            if self.config.apply_noise_reduction:
                p = self._gaussian(p)
            if self.config.apply_truncation_correction and self.config.truncation_width > 0:
                w = self.config.truncation_width
                left = cp.mean(p[:, :10], axis=1, keepdims=True)
                right = cp.mean(p[:, -10:], axis=1, keepdims=True)
                p[:, :w] = left
                p[:, -w:] = right
            if self.config.cosine_weighting:
                p = p * CW
            out[i] = p
        return out


# ---------------------------
# Filtering (FFT ramp)
# ---------------------------

def ramp_flat(m):
    m = int(m // 2)
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
        nProj, nv, nu = stack.shape
        filt_len = 1 << int(np.ceil(np.log2(2 * nu)))
        filter_kernel = get_filter_kernel(self.config.filter_type, filt_len, 1.0)
        start = int(filt_len // 2 - nu // 2)
        end = start + nu
        filtered = cp.zeros_like(stack)
        for i in tqdm(range(nProj), desc="Filtering projections"):
            proj = stack[i]  # (nv, nu)
            fproj = cp.zeros((nv, filt_len), dtype=cp.float32)
            fproj[:, start:end] = proj
            fproj = cp.fft.fft(fproj, axis=1)
            fproj = fproj * filter_kernel[None, :]
            fproj = cp.real(cp.fft.ifft(fproj, axis=1))
            filtered[i] = fproj[:, start:end]
        return filtered


# ---------------------------
# Geometry & Projector separation (attempting to follow ASTRA Toolbox)
# ---------------------------

class CBCTGeometry:
    def __init__(self, config: CBCTConfig):
        self.config = config
        self.derived = config.derived()

    def angles_rad(self) -> np.ndarray:
        start = np.deg2rad(self.config.start_angle)
        step = np.deg2rad(self.config.angle_step)
        return np.array([start + i * step for i in range(self.config.num_projections)], dtype=np.float32)

    def voxel_centers(self):
        return self.derived["xs"], self.derived["ys"], self.derived["zs"]

    def detector_coords(self):
        return self.derived["us"], self.derived["vs"]


# ---------------------------
# Array-sampling backprojector kernel (no texture)
# ---------------------------

BACKPROJ_ARR_KERNEL = r'''
extern "C" {

__global__ void backproj_arr_kernel(const float* __restrict__ proj, float* __restrict__ volume,
                                   const int nx, const int ny, const int nz,
                                   const float voxSizeX, const float voxSizeY, const float voxSizeZ,
                                   const int detCols, const int detRows,
                                   const float detPixelU, const float detPixelV,
                                   const float detOffsetU, const float detOffsetV,
                                   const float sourceDist, const float detectorDist,
                                   const float angle,
                                   const float angleWeight)
    {
        const int ix = blockDim.x * blockIdx.x + threadIdx.x;
        const int iy = blockDim.y * blockIdx.y + threadIdx.y;
        const int iz = blockDim.z * blockIdx.z + threadIdx.z;

        if (ix >= nx || iy >= ny || iz >= nz) return;

        // voxel center in object coordinates
        float vx = (ix - nx / 2.0f + 0.5f) * voxSizeX;
        float vy = (iy - ny / 2.0f + 0.5f) * voxSizeY;
        float vz = (iz - nz / 2.0f + 0.5f) * voxSizeZ;

        float cosA = cosf(angle);
        float sinA = sinf(angle);
        float px = vx * cosA + vy * sinA;
        float py = -vx * sinA + vy * cosA;
        float pz = vz;

        float sourceToVoxelY = py + sourceDist;
        if (sourceToVoxelY <= 0.0f) return;

        float projectionScale = detectorDist / sourceToVoxelY;
        float detU = px * projectionScale - detOffsetU;
        float detV = pz * projectionScale - detOffsetV;

        // convert to pixel indices (0..detCols-1, 0..detRows-1)
        float pixU = detU / detPixelU + detCols / 2.0f;
        float pixV = detV / detPixelV + detRows / 2.0f;

        // bounds check for bilinear interpolation
        if (pixU < 0.0f || pixU >= detCols - 1.0f || pixV < 0.0f || pixV >= detRows - 1.0f) return;

        int u0 = (int)floorf(pixU);
        int v0 = (int)floorf(pixV);
        int u1 = u0 + 1;
        int v1 = v0 + 1;
        float fu = pixU - u0;
        float fv = pixV - v0;

        // row-major projection layout: proj[v * detCols + u]
        int base00 = v0 * detCols + u0;
        float p00 = proj[base00];
        float p01 = proj[v0 * detCols + u1];
        float p10 = proj[v1 * detCols + u0];
        float p11 = proj[v1 * detCols + u1];

        float val = (1.0f - fu) * (1.0f - fv) * p00 +
                    fu * (1.0f - fv) * p01 +
                    (1.0f - fu) * fv * p10 +
                    fu * fv * p11;

        float weight = (sourceDist * sourceDist) / (sourceToVoxelY * sourceToVoxelY);
        float contrib = val * weight * angleWeight;

        int idx = iz * (nx * ny) + iy * nx + ix;
        atomicAdd(&volume[idx], contrib);
    }
}
'''

class CBCTArrayBackprojector:
    def __init__(self, config: CBCTConfig, geometry: CBCTGeometry):
        self.config = config
        self.geometry = geometry
        self.rawmod = cp.RawModule(code=BACKPROJ_ARR_KERNEL, options=("-std=c++11",), name_expressions=["backproj_arr_kernel"])
        self.k_arr = self.rawmod.get_function("backproj_arr_kernel")

    def backproject(self, filtered: cp.ndarray) -> np.ndarray:
        nProj, detRows, detCols = filtered.shape
        nx, ny, nz = self.config.num_voxels

        dx = float(self.config.volume_size_mm[0] / nx)
        dy = float(self.config.volume_size_mm[1] / ny)
        dz = float(self.config.volume_size_mm[2] / nz)

        du = float(self.config.detector_size_mm[0] / detCols)
        dv = float(self.config.detector_size_mm[1] / detRows)

        # Convert detector_offset in mm (or interpret as pixels if flagged)
        # detOffsetU_mm = float(self.config.detector_offset[0])
        # detOffsetV_mm = float(self.config.detector_offset[1])
        # Offset from detector center in mm
        detOffsetU_mm = (float(self.config.detector_offset[0]) - detCols/2.0) * du
        detOffsetV_mm = (float(self.config.detector_offset[1]) - detRows/2.0) * dv
        if self.config.detector_offset_in_pixels:
            detOffsetU_mm = detOffsetU_mm * du
            detOffsetV_mm = detOffsetV_mm * dv

        sourceDist = float(self.config.source_origin_dist)
        detectorDist = float(self.config.source_detector_dist)

        angles = self.geometry.angles_rad()
        angleWeight = 2.0 * np.pi / max(1, nProj)

        # Allocate volume on device
        volume = cp.zeros((nz, ny, nx), dtype=cp.float32)
        vol_ptr = volume.ravel()  # linear memory pointer

        # kernel launch params
        # tune block size for occupancy: (8,8,8) is a reasonable start
        block = (16, 16, 4)
        grid = ((nx + block[0] - 1) // block[0],
                (ny + block[1] - 1) // block[1],
                (nz + block[2] - 1) // block[2])

        logger.info(f"Backprojection: launching {nProj} per-projection kernels (grid={grid}, block={block})")

        for i in tqdm(range(nProj), desc="Backprojecting"):
            proj = filtered[i]  # (nv, nu) on device
            proj = cp.ascontiguousarray(proj.astype(cp.float32))
            proj_buf = proj.ravel()

            try:
                self.k_arr(grid, block,
                           (proj_buf, vol_ptr,
                            np.int32(nx), np.int32(ny), np.int32(nz),
                            np.float32(dx), np.float32(dy), np.float32(dz),
                            np.int32(detCols), np.int32(detRows),
                            np.float32(du), np.float32(dv),
                            np.float32(detOffsetU_mm), np.float32(detOffsetV_mm),
                            np.float32(sourceDist), np.float32(detectorDist),
                            np.float32(float(angles[i])), np.float32(angleWeight)))

            except Exception as e:
                logger.error(f"Array kernel failed for projection {i}: {e}")
                raise

        # Debug output
        vol_np = cp.asnumpy(volume)
        logger.info(f"Volume stats - min: {vol_np.min():.6f}, "
                f"max: {vol_np.max():.6f}, "
                f"mean: {vol_np.mean():.6f}, "
                f"std: {vol_np.std():.6f}")
    
        non_zero = np.count_nonzero(vol_np)
        total = vol_np.size
        logger.info(f"Non-zero voxels: {non_zero}/{total} ({100*non_zero/total:.2f}%)")

        cp.cuda.Device().synchronize()
        vol_np = cp.asnumpy(volume)
        logger.info(f"Backprojection done. Volume shape {vol_np.shape}")
        
        return vol_np


# ---------------------------
# Utilities and pipeline
# ---------------------------

def save_pickle(obj, path):
    import pickle
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def save_config_snapshot(config: CBCTConfig, runtime_info: Dict[str, Any], path: str):
    import pickle
    snapshot = asdict(config)
    snapshot.update(runtime_info)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(snapshot, f)

def run_pipeline(dataset_folder: str, metadata_filename: str = "metadata.json"):

    cfg = CBCTConfig.create_with_metadata(dataset_folder, metadata_filename)
    os.makedirs(cfg.intermediate_path, exist_ok=True)
    os.makedirs(cfg.output_path, exist_ok=True)

    loader = CBCTDataLoader(cfg)
    preproc = CBCTPreprocessor(cfg)
    filt = CBCTRampFilter(cfg)
    geom = CBCTGeometry(cfg)
    backproj = CBCTArrayBackprojector(cfg, geom)

    t0 = time.time()
    projections = loader.load_projection_stack(cfg.input_path, metadata_filename)
    if cfg.save_intermediate:
        save_pickle(cp.asnumpy(projections), os.path.join(cfg.intermediate_path, "projections_raw.pickle"))
    pre = preproc.preprocess_stack(projections)
    if cfg.save_intermediate:
        save_pickle(cp.asnumpy(pre), os.path.join(cfg.intermediate_path, "projections_preprocessed.pickle"))
    filtered = filt.filter_stack(pre)
    if cfg.save_intermediate:
        save_pickle(cp.asnumpy(filtered), os.path.join(cfg.intermediate_path, "projections_filtered.pickle"))
    recon = backproj.backproject(filtered)
    save_pickle(recon, os.path.join(cfg.output_path, "volume.pickle"))
    t1 = time.time()
    logger.info(f"Pipeline complete in {t1 - t0:.2f} s")
    save_config_snapshot(cfg, {"runtime_seconds": t1 - t0}, os.path.join(cfg.output_path, "config_snapshot.pickle"))

    return recon

# ---------------------------
# Command-line
# ---------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    dataset = "data/CBCT_3D_SheppLogan"
    dataset_path = "data/CBCT_3D_SheppLogan/slices"
    ap.add_argument("--input", "-i", required=False, default=dataset, help="Folder with projection files and metadata.json")
    ap.add_argument("--metadata", "-m", default="metadata.json", help="Metadata filename inside input folder")
    args = ap.parse_args()
    recon = run_pipeline(args.input, args.metadata)
    logger.info("Reconstruction finished. Volume stats: min=%.6e max=%.6e mean=%.6e" %
                (float(np.min(recon)), float(np.max(recon)), float(np.mean(recon))))

    from CBCTPipeline_result_view import view_pickled_volume_napari
    view_pickled_volume_napari(path=f'{dataset}/results/volume.pickle')