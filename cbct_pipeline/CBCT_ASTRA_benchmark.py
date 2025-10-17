#!/usr/bin/env python3
"""
CBCT Reconstruction using ASTRA Toolbox with RAW file support
Handles multiple file formats: RAW, TIFF, PNG, JPG
Output matches custom kernel structure
"""

import numpy as np
import os
import time
import logging
import json
from PIL import Image
from tqdm import tqdm
import traceback
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import astra
import pickle
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ASTRACBCTConfig:
    # Dataset parameters
    num_projections: int = 1600
    scan_angle: float = 360.0
    start_angle: float = 270.0
    angle_step: float = 1
    detector_size: Tuple[int, int] = (715, 715)  # (rows, cols)
    pixel_size_u: float = 0.6  # mm
    pixel_size_v: float = 0.6  # mm
    voxel_size: float = 0.006134010138512811  # mm
    source_object_dist: float = 28.625365287711134  # mm
    source_detector_dist: float = 699.9996522369905  # mm
    detector_offset_u: float = 357.5 / 0.01713  # pixels
    detector_offset_v: float = 357.4 / 0.01713  # pixels
    
    # RAW file parameters
    raw_resolution: Tuple[int, int] = (3072, 3072)  # (height, width)
    raw_bit_depth: str = "uint16"
    raw_endianness: str = "little"
    raw_header_size: int = 96  # bytes
    
    # Processing parameters
    dark_current: float = 100.0
    bad_pixel_threshold: int = 32768
    apply_log_correction: bool = True
    apply_bad_pixel_correction: bool = True
    apply_noise_reduction: bool = True
    apply_truncation_correction: bool = True
    truncation_width: int = 125
    
    # Reconstruction parameters
    volume_size: Tuple[int, int, int] = (512, 512, 512)
    algorithm: str = "FDK_CUDA"
    iterations: int = 50
    
    # Memory management
    projection_batch_size: int = 50
    astra_downsample_factor: int = 1  # Placeholder
    
    # IO paths
    input_path: str = "data/slices"
    output_path: str = "data/results_astra"
    
    @classmethod
    def create_with_metadata(cls, folder: str, metadata_filename: str = "metadata.json"):
        """Create config from metadata.json file"""
        cfg = cls()
        cfg.input_path = f"{folder}/slices"
        cfg.output_path = f"{folder}/results_astra"
        
        meta_path = f"{cfg.input_path}/{metadata_filename}"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            cfg._apply_metadata(meta)
            logger.info(f"Loaded metadata from {meta_path}")
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
        if "detector_offset" in meta:
            offsets = tuple(float(x) for x in meta["detector_offset"])
            self.detector_offset_u = offsets[0]
            self.detector_offset_v = offsets[1]
        if "volume_voxels" in meta:
            self.volume_size = tuple(int(x) for x in meta["volume_voxels"])
        if "voxel_size" in meta:
            vs = meta["voxel_size"]
            if isinstance(vs, (list, tuple)):
                self.voxel_size = float(vs[0])
            else:
                self.voxel_size = float(vs)
        if "source_origin_dist" in meta:
            self.source_object_dist = float(meta["source_origin_dist"])
        if "source_detector_dist" in meta:
            self.source_detector_dist = float(meta["source_detector_dist"])
        
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


def discover_projection_files(folder: str, allowed_exts=(".tiff", ".tif", ".jpg", ".jpeg", ".png", ".raw")) -> List[str]:
    """Find all projection files in folder"""
    pattern = os.path.join(folder, "*")
    files = [f for f in sorted(glob.glob(pattern)) 
             if os.path.splitext(f)[1].lower() in allowed_exts]
    return files


class CBCTDataLoader:
    """Load CBCT projection data in multiple formats"""
    
    def __init__(self, config: ASTRACBCTConfig):
        self.config = config
    
    def _load_tiff(self, path: str) -> np.ndarray:
        """Load TIFF file"""
        try:
            import tifffile
            arr = tifffile.imread(path)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            return arr.astype(np.float32)
        except ImportError:
            return np.array(Image.open(path), dtype=np.float32)
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image file (PNG, JPG)"""
        with Image.open(path) as img:
            return np.array(img.convert("F"), dtype=np.float32)
    
    def _load_raw(self, path: str) -> np.ndarray:
        """Load RAW file with metadata-driven parameters"""
        dtype = np.dtype(self.config.raw_bit_depth)
        
        # Handle endianness
        if self.config.raw_endianness == "big":
            dtype = dtype.newbyteorder('>')
        elif self.config.raw_endianness == "little":
            dtype = dtype.newbyteorder('<')
        
        height, width = self.config.raw_resolution
        expected_size = self.config.raw_header_size + (height * width * dtype.itemsize)
        actual_size = os.path.getsize(path)
        
        if actual_size != expected_size:
            logger.warning(f"RAW file size mismatch: expected {expected_size}, got {actual_size}")
        
        with open(path, "rb") as f:
            # Skip header
            f.seek(self.config.raw_header_size)
            data = np.frombuffer(f.read(), dtype=dtype)
        
        try:
            data = data.reshape((height, width))
        except ValueError:
            logger.warning(f"Reshape failed, trying transpose")
            data = data.reshape((width, height)).T
        
        return data.astype(np.float32)
    
    def load_projection_stack(self, folder: Optional[str] = None) -> np.ndarray:
        """Load all projections from folder"""
        if folder is None:
            folder = self.config.input_path
        
        files = discover_projection_files(folder)
        if len(files) == 0:
            raise RuntimeError(f"No projection files found in {folder}")
        
        if len(files) < self.config.num_projections:
            logger.warning(f"Found {len(files)} files but config expects {self.config.num_projections}")
        files = files[:self.config.num_projections]
        
        # Load first projection to determine size
        ext = os.path.splitext(files[0])[1].lower()
        first_proj = self._load_projection(files[0], ext)
        original_h, original_w = first_proj.shape
        
        logger.info(f"Original projection size: {original_h} x {original_w}")
        
        # Apply downsampling
        if self.config.astra_downsample_factor > 1:
            ds = self.config.astra_downsample_factor
            new_h = original_h // ds
            new_w = original_w // ds
            logger.info(f"Downsampling to: {new_h} x {new_w}")
        else:
            new_h, new_w = original_h, original_w
        
        # Allocate array
        projections = np.zeros((self.config.num_projections, new_h, new_w), dtype=np.float32)
        
        # Load in batches
        batch_size = self.config.projection_batch_size
        num_batches = (len(files) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Loading projections"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(files))
            
            for i in range(batch_start, batch_end):
                try:
                    ext = os.path.splitext(files[i])[1].lower()
                    proj = self._load_projection(files[i], ext)
                    
                    # Downsample if needed
                    if self.config.astra_downsample_factor > 1:
                        ds = self.config.astra_downsample_factor
                        proj = proj[::ds, ::ds]
                    
                    projections[i] = proj
                    
                except Exception as e:
                    logger.warning(f"Failed to load {files[i]}: {e}")
                    projections[i] = np.zeros((new_h, new_w), dtype=np.float32)
        
        logger.info(f"Loaded projection stack shape: {projections.shape}")
        return projections
    
    def _load_projection(self, path: str, ext: str) -> np.ndarray:
        """Load single projection based on extension"""
        if ext in (".tif", ".tiff"):
            return self._load_tiff(path)
        elif ext in (".png", ".jpg", ".jpeg"):
            return self._load_image(path)
        elif ext == ".raw":
            return self._load_raw(path)
        else:
            raise RuntimeError(f"Unsupported file format: {ext}")


class CBCTPreprocessor:
    """Preprocess CBCT projections"""
    
    def __init__(self, config: ASTRACBCTConfig):
        self.config = config
    
    def preprocess_projections(self, projections: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline"""
        logger.info("Preprocessing projections...")
        processed = projections.copy()
        
        # Dark current subtraction
        processed = processed - self.config.dark_current
        processed = np.maximum(processed, 1.0)
        
        if self.config.apply_log_correction:
            processed = self._log_correction(processed)
        
        if self.config.apply_bad_pixel_correction:
            processed = self._bad_pixel_correction(processed)
        
        if self.config.apply_noise_reduction:
            processed = self._noise_reduction(processed)
        
        if self.config.apply_truncation_correction:
            processed = self._truncation_correction(processed)
        
        logger.info("Preprocessing complete")
        return processed
    
    def _log_correction(self, projections: np.ndarray) -> np.ndarray:
        """Apply Beer-Lambert logarithmic correction"""
        I0 = np.max(projections, axis=(1, 2), keepdims=True)
        return -np.log(projections / I0)
    
    def _bad_pixel_correction(self, projections: np.ndarray) -> np.ndarray:
        """Correct bad pixels using median filter"""
        try:
            from scipy.ndimage import median_filter
            corrected = projections.copy()
            for i in range(projections.shape[0]):
                bad_mask = projections[i] > self.config.bad_pixel_threshold
                if np.any(bad_mask):
                    filtered = median_filter(projections[i], size=3)
                    corrected[i] = np.where(bad_mask, filtered, projections[i])
            return corrected
        except ImportError:
            logger.warning("scipy not available; skipping bad pixel correction")
            return projections
    
    def _noise_reduction(self, projections: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise reduction"""
        try:
            from scipy.ndimage import gaussian_filter
            filtered = np.zeros_like(projections)
            for i in range(projections.shape[0]):
                filtered[i] = gaussian_filter(projections[i], sigma=1.5)
            return filtered
        except ImportError:
            logger.warning("scipy not available; skipping noise reduction")
            return projections
    
    def _truncation_correction(self, projections: np.ndarray) -> np.ndarray:
        """Correct truncation artifacts"""
        corrected = projections.copy()
        width = self.config.truncation_width
        
        if width > 0 and width < projections.shape[2]:
            for i in range(projections.shape[0]):
                left_edge = np.mean(corrected[i, :, :10], axis=1, keepdims=True)
                corrected[i, :, :width] = left_edge
                right_edge = np.mean(corrected[i, :, -10:], axis=1, keepdims=True)
                corrected[i, :, -width:] = right_edge
        
        return corrected


class ASTRAReconstructor:
    """CBCT reconstruction using ASTRA toolbox"""
    
    def __init__(self, config: ASTRACBCTConfig):
        self.config = config
        self._check_astra()
    
    def _check_astra(self):
        """Check ASTRA installation"""
        logger.info(f"ASTRA version: {astra.__version__}")
        if astra.astra.use_cuda():
            logger.info("CUDA acceleration available")
        else:
            logger.warning("CUDA not available, using CPU")
            if self.config.algorithm.endswith("_CUDA"):
                self.config.algorithm = self.config.algorithm.replace("_CUDA", "")
    
    def create_geometry(self, projections: np.ndarray) -> Tuple[Dict, Dict]:
        """Create ASTRA geometries"""
        num_angles, det_rows, det_cols = projections.shape
        
        angles = np.linspace(
            np.radians(self.config.start_angle),
            np.radians(self.config.start_angle + self.config.scan_angle),
            num_angles,
            endpoint=False
        )
        
        # Adjust for downsampling
        ds = self.config.astra_downsample_factor
        pixel_size_u = self.config.pixel_size_u * ds
        pixel_size_v = self.config.pixel_size_v * ds
        detector_offset_u = self.config.detector_offset_u / ds
        detector_offset_v = self.config.detector_offset_v / ds
        
        proj_geom = astra.create_proj_geom(
            'cone',
            pixel_size_v, pixel_size_u,
            det_rows, det_cols,
            angles,
            self.config.source_object_dist,
            self.config.source_detector_dist - self.config.source_object_dist,
            detector_offset_u - det_cols/2,
            detector_offset_v - det_rows/2
        )
        
        vol_x, vol_y, vol_z = self.config.volume_size
        vol_geom = astra.create_vol_geom(
            vol_y, vol_x, vol_z,
            -vol_x/2 * self.config.voxel_size, vol_x/2 * self.config.voxel_size,
            -vol_y/2 * self.config.voxel_size, vol_y/2 * self.config.voxel_size,
            -vol_z/2 * self.config.voxel_size, vol_z/2 * self.config.voxel_size
        )
        
        return proj_geom, vol_geom
    
    def reconstruct(self, projections: np.ndarray) -> np.ndarray:
        """Perform reconstruction"""
        logger.info(f"Starting ASTRA reconstruction with {self.config.algorithm}")
        
        try:
            proj_geom, vol_geom = self.create_geometry(projections)
            projections = projections.transpose(1, 0, 2)  # (rows, angles, cols)
            
            proj_id = astra.data3d.create('-proj3d', proj_geom, projections)
            vol_id = astra.data3d.create('-vol', vol_geom)
            
            cfg = astra.astra_dict(self.config.algorithm)
            cfg['ReconstructionDataId'] = vol_id
            cfg['ProjectionDataId'] = proj_id
            
            if 'FDK' not in self.config.algorithm:
                cfg['option'] = {'MinConstraint': 0}
                if hasattr(self.config, 'iterations'):
                    cfg['option']['MaxIter'] = self.config.iterations
            
            alg_id = astra.algorithm.create(cfg)
            
            start_time = time.perf_counter()
            if 'FDK' in self.config.algorithm:
                astra.algorithm.run(alg_id)
            else:
                for i in tqdm(range(self.config.iterations), desc="Reconstructing"):
                    astra.algorithm.run(alg_id, 1)
            end_time = time.perf_counter()
            
            reconstruction = astra.data3d.get(vol_id)
            
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(proj_id)
            astra.data3d.delete(vol_id)
            
            logger.info(f"Reconstruction completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Volume shape: {reconstruction.shape}")
            logger.info(f"Value range: [{np.min(reconstruction):.6f}, {np.max(reconstruction):.6f}]")
            
            return reconstruction
            
        except Exception as e:
            logger.error(f"ASTRA reconstruction failed: {e}")
            traceback.print_exc()
            raise


def save_pickle(obj, path):
    """Save object to pickle file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def run_pipeline(dataset_folder: str, metadata_filename: str = "metadata.json"):
    """Run complete ASTRA reconstruction pipeline"""
    
    cfg = ASTRACBCTConfig.create_with_metadata(dataset_folder, metadata_filename)
    os.makedirs(cfg.output_path, exist_ok=True)
    
    loader = CBCTDataLoader(cfg)
    preprocessor = CBCTPreprocessor(cfg)
    reconstructor = ASTRAReconstructor(cfg)
    
    logger.info("=== ASTRA CBCT Reconstruction Pipeline ===")
    
    t0 = time.time()
    
    # Load projections
    projections = loader.load_projection_stack()
    
    # Preprocess
    projections_processed = preprocessor.preprocess_projections(projections)
    
    # Reconstruct
    reconstruction = reconstructor.reconstruct(projections_processed)
    
    # Save results (matching custom kernel output)
    save_pickle(reconstruction, os.path.join(cfg.output_path, "volume.pickle"))
    
    t1 = time.time()
    logger.info(f"Pipeline complete in {t1 - t0:.2f} s")
    
    # Save config snapshot
    config_data = {
        "runtime_seconds": t1 - t0,
        "algorithm": cfg.algorithm,
        "volume_shape": reconstruction.shape,
        "voxel_size": cfg.voxel_size,
        "astra_downsample_factor": cfg.astra_downsample_factor
    }
    save_pickle(config_data, os.path.join(cfg.output_path, "config_snapshot.pickle"))
    
    logger.info("Reconstruction finished. Volume stats: min=%.6e max=%.6e mean=%.6e" %
                (float(np.min(reconstruction)), float(np.max(reconstruction)), float(np.mean(reconstruction))))
    
    return reconstruction


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="ASTRA CBCT Reconstruction with RAW file support")
    ap.add_argument("--input", "-i", default="data/20240530_ITRI_downsampled_4x",
                    help="Dataset folder containing slices/ and metadata.json")
    ap.add_argument("--metadata", "-m", default="metadata.json",
                    help="Metadata filename")
    args = ap.parse_args()
    
    try:
        recon = run_pipeline(args.input, args.metadata)
        logger.info("Success! View results with CBCTPipeline_result_view.py")
        from CBCTPipeline_result_view import view_pickled_volume_napari
        view_pickled_volume_napari(os.path.join(args.input, "results_astra", "volume.pickle"))
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)