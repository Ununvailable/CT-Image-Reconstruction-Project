#!/usr/bin/env python3
"""
CBCT Reconstruction using ASTRA Toolbox with RAW file support
Handles multiple file formats: RAW, TIFF, PNG, JPG
Supports anisotropic voxels and flexible geometry configuration
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
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List, Union
import astra
import pickle
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ASTRACBCTConfig:
    """Configuration for ASTRA CBCT reconstruction pipeline"""
    
    # Acquisition geometry
    acquisition_num_projections: int = 360
    acquisition_scan_range_deg: float = 360.0
    acquisition_start_angle_deg: float = 0.0
    acquisition_angle_step_deg: float = 1.0
    acquisition_rotation_direction: str = "CCW"
    
    # Source-detector geometry
    source_to_origin_dist_mm: float = 110.0
    source_to_detector_dist_mm: float = 757.0
    source_magnification: float = 6.88
    
    # Detector parameters
    detector_rows_px: int = 768
    detector_cols_px: int = 768
    detector_pixel_pitch_u_mm: float = 0.56
    detector_pixel_pitch_v_mm: float = 0.56
    detector_physical_size_u_mm: float = 430.0
    detector_physical_size_v_mm: float = 430.0
    
    # Volume parameters
    volume_nx_vox: int = 256
    volume_ny_vox: int = 256
    volume_nz_vox: int = 256
    volume_size_x_mm: float = 180.0
    volume_size_y_mm: float = 180.0
    volume_size_z_mm: float = 180.0
    volume_voxel_pitch_mm: float = 0.703125
    
    # Preprocessing parameters
    preprocessing_downsample_factor: int = 1
    preprocessing_apply_log: bool = False
    preprocessing_apply_bad_pixel_correction: bool = False
    preprocessing_apply_gaussian_filter: bool = False
    preprocessing_gaussian_sigma_px: float = 1.5
    preprocessing_dark_current: float = 0.0
    preprocessing_bad_pixel_threshold: int = 32768
    
    # RAW file parameters
    raw_rows_px: int = 1536
    raw_cols_px: int = 1536
    raw_dtype: str = "uint16"
    raw_endian: str = "little"
    raw_header_bytes: int = 0
    
    # Calibration parameters
    calibration_offset_u_px: float = 0.0
    calibration_offset_v_px: float = 0.0
    calibration_offset_reference: str = "fullres"
    calibration_tilt_deg: float = 0.0
    calibration_skew_deg: float = 0.0
    calibration_sod_correction_mm: float = 0.0
    calibration_angular_offset_deg: float = 0.0
    
    # Reconstruction parameters
    reconstruction_algorithm: str = "FDK_CUDA"
    reconstruction_iterations: int = 50
    
    # I/O parameters
    input_path: str = "slices/"
    output_path: str = "results_astra/"
    file_format: str = "tiff"
    
    @classmethod
    def create_with_metadata(cls, folder: str, metadata_filename: str = "metadata.json"):
        """Create configuration from metadata.json file"""
        cfg = cls()
        cfg.input_path = os.path.join(folder, "slices")
        cfg.output_path = os.path.join(folder, "results_astra")
        
        meta_path = os.path.join(cfg.input_path, metadata_filename)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cfg._apply_metadata(meta)
            logger.info(f"Loaded metadata from {meta_path}")
        else:
            logger.warning(f"No metadata file at {meta_path}, using defaults")
        
        return cfg
    
    def _apply_metadata(self, meta: Dict[str, Any]):
        """Apply metadata fields to configuration"""
        
        # Acquisition parameters
        if "acquisition_num_projections" in meta:
            self.acquisition_num_projections = int(meta["acquisition_num_projections"])
        if "acquisition_angle_step_deg" in meta:
            self.acquisition_angle_step_deg = float(meta["acquisition_angle_step_deg"])
        if "acquisition_start_angle_deg" in meta:
            self.acquisition_start_angle_deg = float(meta["acquisition_start_angle_deg"])
        if "acquisition_scan_range_deg" in meta:
            self.acquisition_scan_range_deg = float(meta["acquisition_scan_range_deg"])
        if "acquisition_rotation_direction" in meta:
            self.acquisition_rotation_direction = str(meta["acquisition_rotation_direction"])
        
        # Source-detector geometry
        if "source_to_origin_dist_mm" in meta:
            self.source_to_origin_dist_mm = float(meta["source_to_origin_dist_mm"])
        if "source_to_detector_dist_mm" in meta:
            self.source_to_detector_dist_mm = float(meta["source_to_detector_dist_mm"])
        if "source_magnification" in meta:
            self.source_magnification = float(meta["source_magnification"])
        
        # Detector parameters
        if "detector_rows_px" in meta:
            self.detector_rows_px = int(meta["detector_rows_px"])
        if "detector_cols_px" in meta:
            self.detector_cols_px = int(meta["detector_cols_px"])
        if "detector_pixel_pitch_u_mm" in meta:
            self.detector_pixel_pitch_u_mm = float(meta["detector_pixel_pitch_u_mm"])
        if "detector_pixel_pitch_v_mm" in meta:
            self.detector_pixel_pitch_v_mm = float(meta["detector_pixel_pitch_v_mm"])
        if "detector_physical_size_u_mm" in meta:
            self.detector_physical_size_u_mm = float(meta["detector_physical_size_u_mm"])
        if "detector_physical_size_v_mm" in meta:
            self.detector_physical_size_v_mm = float(meta["detector_physical_size_v_mm"])
        
        # Volume parameters
        if "volume_nx_vox" in meta:
            self.volume_nx_vox = int(meta["volume_nx_vox"])
        if "volume_ny_vox" in meta:
            self.volume_ny_vox = int(meta["volume_ny_vox"])
        if "volume_nz_vox" in meta:
            self.volume_nz_vox = int(meta["volume_nz_vox"])
        if "volume_size_x_mm" in meta:
            self.volume_size_x_mm = float(meta["volume_size_x_mm"])
        if "volume_size_y_mm" in meta:
            self.volume_size_y_mm = float(meta["volume_size_y_mm"])
        if "volume_size_z_mm" in meta:
            self.volume_size_z_mm = float(meta["volume_size_z_mm"])
        if "volume_voxel_pitch_mm" in meta:
            self.volume_voxel_pitch_mm = float(meta["volume_voxel_pitch_mm"])
        
        # Preprocessing parameters
        if "preprocessing_downsample_factor" in meta:
            self.preprocessing_downsample_factor = int(meta["preprocessing_downsample_factor"])
        if "preprocessing_apply_log" in meta:
            self.preprocessing_apply_log = bool(meta["preprocessing_apply_log"])
        if "preprocessing_apply_bad_pixel_correction" in meta:
            self.preprocessing_apply_bad_pixel_correction = bool(meta["preprocessing_apply_bad_pixel_correction"])
        if "preprocessing_apply_gaussian_filter" in meta:
            self.preprocessing_apply_gaussian_filter = bool(meta["preprocessing_apply_gaussian_filter"])
        if "preprocessing_gaussian_sigma_px" in meta:
            self.preprocessing_gaussian_sigma_px = float(meta["preprocessing_gaussian_sigma_px"])
        if "preprocessing_dark_current" in meta:
            self.preprocessing_dark_current = float(meta["preprocessing_dark_current"])
        
        # RAW file parameters
        if "raw_rows_px" in meta:
            self.raw_rows_px = int(meta["raw_rows_px"])
        if "raw_cols_px" in meta:
            self.raw_cols_px = int(meta["raw_cols_px"])
        if "raw_dtype" in meta:
            self.raw_dtype = str(meta["raw_dtype"])
        if "raw_endian" in meta:
            self.raw_endian = str(meta["raw_endian"]).lower()
        if "raw_header_bytes" in meta:
            self.raw_header_bytes = int(meta["raw_header_bytes"])
        
        # Calibration parameters
        if "calibration_offset_u_px" in meta:
            self.calibration_offset_u_px = float(meta["calibration_offset_u_px"])
        if "calibration_offset_v_px" in meta:
            self.calibration_offset_v_px = float(meta["calibration_offset_v_px"])
        if "calibration_offset_reference" in meta:
            self.calibration_offset_reference = str(meta["calibration_offset_reference"])
        if "calibration_tilt_deg" in meta:
            self.calibration_tilt_deg = float(meta["calibration_tilt_deg"])
        if "calibration_skew_deg" in meta:
            self.calibration_skew_deg = float(meta["calibration_skew_deg"])
        if "calibration_sod_correction_mm" in meta:
            self.calibration_sod_correction_mm = float(meta["calibration_sod_correction_mm"])
        if "calibration_angular_offset_deg" in meta:
            self.calibration_angular_offset_deg = float(meta["calibration_angular_offset_deg"])
        
        # Reconstruction parameters
        if "reconstruction_algorithm" in meta:
            self.reconstruction_algorithm = str(meta["reconstruction_algorithm"])
        if "reconstruction_iterations" in meta:
            self.reconstruction_iterations = int(meta["reconstruction_iterations"])
        
        # File format
        if "file_format" in meta:
            self.file_format = str(meta["file_format"])
        
        # Log loaded configuration
        logger.info(f"Loaded: {self.acquisition_num_projections} projections, "
                   f"detector {self.detector_rows_px}×{self.detector_cols_px}, "
                   f"SOD={self.source_to_origin_dist_mm}mm, SDD={self.source_to_detector_dist_mm}mm")
        logger.info(f"Calibration: offset_u={self.calibration_offset_u_px}px, "
                   f"offset_v={self.calibration_offset_v_px}px, "
                   f"tilt={self.calibration_tilt_deg}deg")
    
    def get_voxel_sizes(self) -> Tuple[float, float, float]:
        """Get voxel sizes as tuple (X, Y, Z)"""
        return (self.volume_voxel_pitch_mm, self.volume_voxel_pitch_mm, self.volume_voxel_pitch_mm)
    
    def get_volume_size(self) -> Tuple[int, int, int]:
        """Get volume dimensions as tuple (X, Y, Z)"""
        return (self.volume_nx_vox, self.volume_ny_vox, self.volume_nz_vox)
    
    def get_detector_size(self) -> Tuple[int, int]:
        """Get detector dimensions as tuple (rows, cols)"""
        return (self.detector_rows_px, self.detector_cols_px)
    
    def get_raw_resolution(self) -> Tuple[int, int]:
        """Get RAW file resolution as tuple (rows, cols)"""
        return (self.raw_rows_px, self.raw_cols_px)
    
    def get_calibration_offset_px(self) -> Tuple[float, float]:
        """
        Get calibration offset in pixels at current working resolution.
        Converts from fullres if necessary.
        """
        offset_u = self.calibration_offset_u_px
        offset_v = self.calibration_offset_v_px
        
        if self.calibration_offset_reference != "fullres":
            ds = self.preprocessing_downsample_factor
            offset_u = offset_u / ds
            offset_v = offset_v / ds
        
        return (offset_u, offset_v)


def discover_projection_files(folder: str, file_format: str = None, allowed_exts=None) -> List[str]:
    """Find projection files in folder"""
    if file_format and file_format.lower() == "raw":
        pattern = os.path.join(folder, "*.raw")
        files = sorted(glob.glob(pattern))
    elif allowed_exts:
        pattern = os.path.join(folder, "*")
        files = [f for f in sorted(glob.glob(pattern)) 
                if os.path.splitext(f)[1].lower() in allowed_exts]
    else:
        default_exts = (".tiff", ".tif", ".jpg", ".jpeg", ".png", ".raw")
        pattern = os.path.join(folder, "*")
        files = [f for f in sorted(glob.glob(pattern)) 
                if os.path.splitext(f)[1].lower() in default_exts]
    
    if len(files) == 0:
        logger.warning(f"No projection files found in {folder}")
    
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
        """Load PNG/JPG image file"""
        with Image.open(path) as img:
            return np.array(img.convert("F"), dtype=np.float32)
    
    def _load_raw(self, path: str) -> np.ndarray:
        """Load RAW binary file with optional header skip and endianness control"""
        base_dtype = np.dtype(self.config.raw_dtype)
        
        if self.config.raw_endian == "little":
            dtype = base_dtype.newbyteorder('<')
        elif self.config.raw_endian == "big":
            dtype = base_dtype.newbyteorder('>')
        else:
            dtype = base_dtype
        
        height, width = self.config.get_raw_resolution()
        header_bytes = self.config.raw_header_bytes
        
        expected_data_size = height * width * dtype.itemsize
        expected_total_size = header_bytes + expected_data_size
        actual_size = os.path.getsize(path)
        
        if actual_size < expected_total_size:
            logger.warning(f"File {os.path.basename(path)} smaller than expected, recalculating header")
            header_bytes = actual_size - expected_data_size
            if header_bytes < 0:
                raise ValueError(f"File size {actual_size} incompatible with resolution {height}×{width}")
            logger.info(f"Adjusted header to {header_bytes} bytes")
        
        with open(path, "rb") as f:
            if header_bytes > 0:
                f.seek(header_bytes)
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
        
        files = discover_projection_files(folder, file_format=self.config.file_format)
        
        if len(files) == 0:
            raise RuntimeError(f"No projection files found in {folder}")
        
        num_proj = self.config.acquisition_num_projections
        if len(files) < num_proj:
            logger.warning(f"Found {len(files)} files but config expects {num_proj}")
        
        files = files[:num_proj]
        
        projections = []
        ds = self.config.preprocessing_downsample_factor
        
        for path in tqdm(files, desc="Loading projections"):
            try:
                ext = os.path.splitext(path)[1].lower()
                
                if ext in (".tif", ".tiff"):
                    proj = self._load_tiff(path)
                elif ext in (".png", ".jpg", ".jpeg"):
                    proj = self._load_image(path)
                elif ext == ".raw":
                    proj = self._load_raw(path)
                else:
                    raise RuntimeError(f"Unsupported file format: {ext}")
                
                # Downsample if needed
                if ds > 1:
                    proj = proj[::ds, ::ds]
                
                projections.append(proj)
                
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                raise
        
        stack = np.stack(projections, axis=0).astype(np.float32)
        logger.info(f"Loaded projection stack shape: {stack.shape}")
        logger.info(f"Projection value range: [{stack.min():.3f}, {stack.max():.3f}]")
        
        return stack


class CBCTPreprocessor:
    """Preprocess CBCT projections"""
    
    def __init__(self, config: ASTRACBCTConfig):
        self.config = config
    
    def preprocess_projections(self, projections: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline"""
        logger.info("Preprocessing projections...")
        processed = projections.copy()
        
        # Dark current subtraction
        if self.config.preprocessing_dark_current != 0:
            processed = processed - self.config.preprocessing_dark_current
            processed = np.maximum(processed, 1.0)
        
        if self.config.preprocessing_apply_log:
            processed = self._log_correction(processed)
        
        if self.config.preprocessing_apply_bad_pixel_correction:
            processed = self._bad_pixel_correction(processed)
        
        if self.config.preprocessing_apply_gaussian_filter:
            processed = self._noise_reduction(processed)
        
        logger.info(f"Preprocessed value range: [{processed.min():.3f}, {processed.max():.3f}]")
        logger.info("Preprocessing complete")
        
        return processed
    
    def _log_correction(self, projections: np.ndarray) -> np.ndarray:
        """Apply Beer-Lambert logarithmic correction"""
        I0 = np.max(projections, axis=(1, 2), keepdims=True)
        return -np.log(np.maximum(projections / I0, 1e-10))
    
    def _bad_pixel_correction(self, projections: np.ndarray) -> np.ndarray:
        """Correct bad pixels using median filter"""
        try:
            from scipy.ndimage import median_filter
            corrected = projections.copy()
            for i in range(projections.shape[0]):
                bad_mask = projections[i] > self.config.preprocessing_bad_pixel_threshold
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
            sigma = self.config.preprocessing_gaussian_sigma_px
            filtered = np.zeros_like(projections)
            for i in range(projections.shape[0]):
                filtered[i] = gaussian_filter(projections[i], sigma=sigma)
            return filtered
        except ImportError:
            logger.warning("scipy not available; skipping noise reduction")
            return projections


class ASTRAReconstructor:
    """CBCT reconstruction using ASTRA Toolbox"""
    
    def __init__(self, config: ASTRACBCTConfig):
        self.config = config
        self._check_astra()
    
    def _check_astra(self):
        """Check ASTRA installation and CUDA availability"""
        logger.info(f"ASTRA Toolbox version: {astra.__version__}")
        
        if astra.astra.use_cuda():
            logger.info("CUDA acceleration: AVAILABLE")
        else:
            logger.warning("CUDA acceleration: NOT AVAILABLE (using CPU)")
            if self.config.reconstruction_algorithm.endswith("_CUDA"):
                self.config.reconstruction_algorithm = self.config.reconstruction_algorithm.replace("_CUDA", "")
                logger.info(f"Switched algorithm to {self.config.reconstruction_algorithm}")
    
    def create_geometry(self, projections: np.ndarray) -> Tuple[Dict, Dict]:
        """Create ASTRA projection and volume geometries using cone_vec"""
        num_angles, det_rows, det_cols = projections.shape
        
        # Generate angles with calibration offset
        start_angle = self.config.acquisition_start_angle_deg + self.config.calibration_angular_offset_deg
        scan_range = self.config.acquisition_scan_range_deg
        
        angles = np.linspace(
            np.radians(start_angle),
            np.radians(start_angle + scan_range),
            num_angles,
            endpoint=False
        )
        
        # Handle rotation direction
        if self.config.acquisition_rotation_direction.upper() == "CW":
            angles = -angles
        
        # Pixel sizes at working resolution (after downsampling)
        ds = self.config.preprocessing_downsample_factor
        pixel_pitch_u = self.config.detector_pixel_pitch_u_mm * ds
        pixel_pitch_v = self.config.detector_pixel_pitch_v_mm * ds
        
        # Get calibration offsets in working resolution pixels, then convert to mm
        offset_u_px, offset_v_px = self.config.get_calibration_offset_px()
        offset_u_mm = offset_u_px * pixel_pitch_u
        offset_v_mm = offset_v_px * pixel_pitch_v
        
        # Source-object and object-detector distances with SOD correction
        SOD = self.config.source_to_origin_dist_mm + self.config.calibration_sod_correction_mm
        ODD = self.config.source_to_detector_dist_mm - SOD
        
        # Detector tilt and skew
        tilt_rad = np.radians(self.config.calibration_tilt_deg)
        skew_rad = np.radians(self.config.calibration_skew_deg)
        
        # Build vectors for each angle
        vectors = np.zeros((num_angles, 12))
        for i, ang in enumerate(angles):
            # Source position
            vectors[i, 0] = np.sin(ang) * SOD
            vectors[i, 1] = -np.cos(ang) * SOD
            vectors[i, 2] = 0
            
            # Detector center position (with offset)
            vectors[i, 3] = -np.sin(ang) * ODD + np.cos(ang) * offset_u_mm
            vectors[i, 4] = np.cos(ang) * ODD + np.sin(ang) * offset_u_mm
            vectors[i, 5] = offset_v_mm
            
            # Detector u-axis (horizontal) with tilt
            vectors[i, 6] = np.cos(ang) * np.cos(tilt_rad) * pixel_pitch_u
            vectors[i, 7] = np.sin(ang) * np.cos(tilt_rad) * pixel_pitch_u
            vectors[i, 8] = np.sin(tilt_rad) * pixel_pitch_u
            
            # Detector v-axis (vertical) with tilt and skew
            vectors[i, 9] = -np.sin(tilt_rad) * np.cos(ang) * pixel_pitch_v + np.sin(skew_rad) * np.cos(ang) * pixel_pitch_v
            vectors[i, 10] = -np.sin(tilt_rad) * np.sin(ang) * pixel_pitch_v + np.sin(skew_rad) * np.sin(ang) * pixel_pitch_v
            vectors[i, 11] = np.cos(tilt_rad) * pixel_pitch_v
        
        proj_geom = astra.create_proj_geom('cone_vec', det_rows, det_cols, vectors)
        
        # Volume geometry
        vol_x, vol_y, vol_z = self.config.get_volume_size()
        voxel_x, voxel_y, voxel_z = self.config.get_voxel_sizes()
        
        vol_geom = astra.create_vol_geom(
            vol_y, vol_x, vol_z,
            -vol_x/2 * voxel_x, vol_x/2 * voxel_x,
            -vol_y/2 * voxel_y, vol_y/2 * voxel_y,
            -vol_z/2 * voxel_z, vol_z/2 * voxel_z
        )
        
        logger.info(f"Projection geometry: {num_angles} angles, {det_rows}×{det_cols} detector")
        logger.info(f"Volume geometry: {vol_x}×{vol_y}×{vol_z} voxels, "
                   f"{vol_x*voxel_x:.2f}×{vol_y*voxel_y:.2f}×{vol_z*voxel_z:.2f} mm")
        logger.info(f"Effective offsets: u={offset_u_mm:.3f}mm, v={offset_v_mm:.3f}mm")
        
        return proj_geom, vol_geom
    
    def reconstruct(self, projections: np.ndarray) -> np.ndarray:
        """Perform 3D reconstruction"""
        logger.info(f"Starting reconstruction with {self.config.reconstruction_algorithm}")
        
        try:
            # Create geometries
            proj_geom, vol_geom = self.create_geometry(projections)
            
            # Transpose projections for ASTRA format: (angles, rows, cols) -> (rows, angles, cols)
            projections = projections.transpose(1, 0, 2)
            
            # Create ASTRA data objects
            proj_id = astra.data3d.create('-proj3d', proj_geom, projections)
            vol_id = astra.data3d.create('-vol', vol_geom)
            
            # Configure reconstruction algorithm
            cfg = astra.astra_dict(self.config.reconstruction_algorithm)
            cfg['ReconstructionDataId'] = vol_id
            cfg['ProjectionDataId'] = proj_id
            
            # Add options for iterative algorithms
            if 'FDK' not in self.config.reconstruction_algorithm:
                cfg['option'] = {'MinConstraint': 0}
                if self.config.reconstruction_iterations > 0:
                    cfg['option']['MaxIter'] = self.config.reconstruction_iterations
            
            alg_id = astra.algorithm.create(cfg)
            
            # Run reconstruction
            start_time = time.perf_counter()
            
            if 'FDK' in self.config.reconstruction_algorithm:
                astra.algorithm.run(alg_id)
            else:
                for i in tqdm(range(self.config.reconstruction_iterations), desc="Reconstructing"):
                    astra.algorithm.run(alg_id, 1)
            
            end_time = time.perf_counter()
            
            # Retrieve reconstruction
            reconstruction = astra.data3d.get(vol_id)
            
            # Cleanup ASTRA objects
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(proj_id)
            astra.data3d.delete(vol_id)
            
            logger.info(f"Reconstruction completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Volume shape: {reconstruction.shape}")
            logger.info(f"Value range: [{np.min(reconstruction):.6f}, {np.max(reconstruction):.6f}]")
            logger.info(f"Mean value: {np.mean(reconstruction):.6f}")
            
            return reconstruction
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            traceback.print_exc()
            raise


def save_pickle(obj, path):
    """Save object to pickle file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved to {path}")


def run_pipeline(dataset_folder: str, metadata_filename: str = "metadata.json"):
    """Run complete CBCT reconstruction pipeline"""
    
    logger.info("="*60)
    logger.info("ASTRA CBCT Reconstruction Pipeline")
    logger.info("="*60)
    
    # Load configuration
    cfg = ASTRACBCTConfig.create_with_metadata(dataset_folder, metadata_filename)
    os.makedirs(cfg.output_path, exist_ok=True)
    
    # Initialize pipeline components
    loader = CBCTDataLoader(cfg)
    preprocessor = CBCTPreprocessor(cfg)
    reconstructor = ASTRAReconstructor(cfg)
    
    t0 = time.time()
    
    # Load projections
    projections = loader.load_projection_stack()
    
    # Preprocess projections
    projections_processed = preprocessor.preprocess_projections(projections)
    
    # Reconstruct volume
    reconstruction = reconstructor.reconstruct(projections_processed)
    
    # Save results
    save_pickle(reconstruction, os.path.join(cfg.output_path, "volume.pickle"))
    
    # Save configuration snapshot
    config_snapshot = {
        "runtime_seconds": time.time() - t0,
        "algorithm": cfg.reconstruction_algorithm,
        "volume_shape": reconstruction.shape,
        "voxel_size_mm": cfg.get_voxel_sizes(),
        "num_projections": cfg.acquisition_num_projections,
        "detector_size": cfg.get_detector_size(),
        "downsample_factor": cfg.preprocessing_downsample_factor,
        "calibration_offset_u_px": cfg.calibration_offset_u_px,
        "calibration_offset_v_px": cfg.calibration_offset_v_px,
        "calibration_tilt_deg": cfg.calibration_tilt_deg
    }
    save_pickle(config_snapshot, os.path.join(cfg.output_path, "config_snapshot.pickle"))
    
    t1 = time.time()
    
    logger.info("="*60)
    logger.info(f"Pipeline completed in {t1 - t0:.2f} seconds")
    logger.info(f"Volume statistics:")
    logger.info(f"  Min:  {float(np.min(reconstruction)):.6e}")
    logger.info(f"  Max:  {float(np.max(reconstruction)):.6e}")
    logger.info(f"  Mean: {float(np.mean(reconstruction)):.6e}")
    logger.info(f"  Std:  {float(np.std(reconstruction)):.6e}")
    logger.info("="*60)
    
    return reconstruction


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ASTRA CBCT Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python %(prog)s --input data/my_scan
    python %(prog)s --input data/my_scan --metadata config.json
        """
    )
    
    parser.add_argument("--input", "-i", 
                       default="data/20251119_Tako_IronWire",
                       help="Dataset folder containing slices/ and metadata.json")
    parser.add_argument("--metadata", "-m", 
                       default="metadata.json",
                       help="Metadata filename (default: metadata.json)")
    
    args = parser.parse_args()
    
    try:
        recon = run_pipeline(args.input, args.metadata)
        
        logger.info("Reconstruction successful!")
        logger.info(f"Results saved to: {os.path.join(args.input, 'results_astra')}")
        
        # Try to launch viewer if available
        try:
            from CBCTPipeline_result_view import view_pickled_volume_napari
            logger.info("Launching napari viewer...")
            view_pickled_volume_napari(os.path.join(args.input, "results_astra", "volume.pickle"))
        except ImportError:
            logger.info("napari viewer not available. View results manually.")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)