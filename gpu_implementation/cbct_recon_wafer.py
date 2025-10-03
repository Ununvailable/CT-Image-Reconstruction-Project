#!/usr/bin/env python3
"""
CBCT Reconstruction for SiC wafer dataset using ASTRA Toolbox
- Input: raw projections (16-bit unsigned, little-endian, 96-byte header)
- Output: 3D volume (nii.gz + preview slices)
"""

import numpy as np
import os
import time
import logging
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import astra
import traceback
import sys
import nibabel as nib
from PIL import Image

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ========================
# Config
# ========================
@dataclass
class ASTRACBCTConfig:
    # Dataset parameters
    num_projections: int = 721
    scan_angle: float = 360.0
    start_angle: float = 0.0
    detector_size: Tuple[int, int] = (3072, 3072)  # (rows, cols)
    pixel_size_u: float = 0.153  # mm
    pixel_size_v: float = 0.153  # mm
    voxel_size: float = 0.1      # mm (cover ~180 mm with 1024 voxels)
    source_object_dist: float = 300.0   # mm
    source_detector_dist: float = 1200.0  # mm
    detector_offset_u: float = 0.0
    detector_offset_v: float = 0.0

    # Processing
    dark_current: float = 100.0
    apply_log_correction: bool = True
    apply_bad_pixel_correction: bool = False
    apply_noise_reduction: bool = False

    # Reconstruction
    volume_size: Tuple[int, int, int] = (1024, 1024, 512)
    algorithm: str = "FDK_CUDA"

    # Memory
    projection_batch_size: int = 50


# ========================
# Loader
# ========================
class CBCTDataLoader:
    def __init__(self, config: ASTRACBCTConfig):
        self.config = config

    def load_projections(self, proj_folder: str) -> np.ndarray:
        rows, cols = self.config.detector_size
        projections = np.zeros((self.config.num_projections, rows, cols), dtype=np.float32)

        logger.info(f"Loading {self.config.num_projections} raw projections from {proj_folder}")

        for i in tqdm(range(self.config.num_projections), desc="Loading RAW projections"):
            raw_file = os.path.join(proj_folder, f"{i+1}.RAW")
            with open(raw_file, "rb") as f:
                f.seek(0)  # skip header
                data = np.fromfile(f, dtype="<u2", count=rows*cols)
                proj = data.reshape((rows, cols))
                projections[i] = proj

        logger.info(f"Loaded projections shape: {projections.shape}")
        return projections


# ========================
# Preprocessor
# ========================
class CBCTPreprocessor:
    def __init__(self, config: ASTRACBCTConfig):
        self.config = config

    def preprocess(self, projections: np.ndarray) -> np.ndarray:
        logger.info("Preprocessing projections...")

        proc = projections.copy()
        proc = proc - self.config.dark_current
        proc = np.maximum(proc, 1.0)

        if self.config.apply_log_correction:
            I0 = np.max(proc, axis=(1, 2), keepdims=True)
            proc = -np.log(proc / I0)

        logger.info("Preprocessing done.")
        return proc


# ========================
# Reconstructor
# ========================
class ASTRAReconstructor:
    def __init__(self, config: ASTRACBCTConfig):
        self.config = config

    def create_geometry(self, projections: np.ndarray):
        num_angles, det_rows, det_cols = projections.shape

        angles = np.linspace(
            np.radians(self.config.start_angle),
            np.radians(self.config.start_angle + self.config.scan_angle),
            num_angles,
            endpoint=False
        )

        proj_geom = astra.create_proj_geom(
            "cone",
            self.config.pixel_size_v,
            self.config.pixel_size_u,
            det_rows,
            det_cols,
            angles,
            self.config.source_object_dist,
            self.config.source_detector_dist - self.config.source_object_dist,
            self.config.detector_offset_u,
            self.config.detector_offset_v
        )

        vol_x, vol_y, vol_z = self.config.volume_size
        vol_geom = astra.create_vol_geom(
            vol_y, vol_x, vol_z,
            -vol_x/2*self.config.voxel_size, vol_x/2*self.config.voxel_size,
            -vol_y/2*self.config.voxel_size, vol_y/2*self.config.voxel_size,
            -vol_z/2*self.config.voxel_size, vol_z/2*self.config.voxel_size
        )

        return proj_geom, vol_geom

    def reconstruct(self, projections: np.ndarray):
        proj_geom, vol_geom = self.create_geometry(projections)
        projections = projections.transpose(1, 0, 2)  # (rows, angles, cols)

        proj_id = astra.data3d.create("-proj3d", proj_geom, projections)
        vol_id = astra.data3d.create("-vol", vol_geom)

        cfg = astra.astra_dict(self.config.algorithm)
        cfg["ReconstructionDataId"] = vol_id
        cfg["ProjectionDataId"] = proj_id

        alg_id = astra.algorithm.create(cfg)

        start = time.time()
        astra.algorithm.run(alg_id)
        end = time.time()

        recon = astra.data3d.get(vol_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(proj_id)
        astra.data3d.delete(vol_id)

        logger.info(f"Reconstruction completed in {end-start:.2f} sec")
        return recon


# ========================
# Save results
# ========================
def save_results(recon: np.ndarray, voxel_size: float, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    # Save as nii.gz
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1])
    nii = nib.Nifti1Image(recon.astype(np.float32), affine)
    nii_path = os.path.join(output_folder, "recon.nii.gz")
    nib.save(nii, nii_path)
    logger.info(f"Saved NIfTI volume: {nii_path}")

    # Save previews
    midz, midy, midx = np.array(recon.shape)//2
    slices = {
        "axial.png": recon[midz],
        "coronal.png": recon[:, midy, :],
        "sagittal.png": recon[:, :, midx],
    }
    for name, sl in slices.items():
        img = (sl - sl.min()) / (np.ptp(sl) + 1e-6)
        img = Image.fromarray((img*255).astype(np.uint8))
        img.save(os.path.join(output_folder, name))
        logger.info(f"Saved preview: {name}")


# ========================
# Main
# ========================
def main():
    config = ASTRACBCTConfig()
    proj_folder = "20250922_InnoCare/0922SSiC/720"
    out_folder = "data/recon_output"

    try:
        loader = CBCTDataLoader(config)
        pre = CBCTPreprocessor(config)
        reconstructor = ASTRAReconstructor(config)

        proj = loader.load_projections(proj_folder)
        proj_proc = pre.preprocess(proj)
        recon = reconstructor.reconstruct(proj_proc)
        save_results(recon, config.voxel_size, out_folder)

    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
