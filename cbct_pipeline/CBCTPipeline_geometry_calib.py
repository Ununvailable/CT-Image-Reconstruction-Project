#!/usr/bin/env python3
"""
Comprehensive CT Geometry Calibration Suite
Multi-parameter geometry estimation for CBCT reconstruction

Estimates and refines:
- Center of Rotation (COR) offset (u and v)
- Detector tilt (rotation around beam axis)
- Detector skew (non-orthogonality of u/v axes)
- Source-Object Distance (SOD) error
- Angular offset (projection indexing phase)

Author: [Your name]
Date: 2025
"""

import numpy as np
import os
import glob
import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any, Callable
from enum import Enum
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# GPU ACCELERATION (CuPy)
# =============================================================================

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.ndimage import shift as gpu_shift
    from cupyx.scipy.ndimage import rotate as gpu_rotate
    from cupyx.scipy.ndimage import zoom as gpu_zoom
    from cupyx.scipy.ndimage import affine_transform as gpu_affine_transform
    GPU_AVAILABLE = True
    # logger.info(f"CuPy GPU acceleration: AVAILABLE (Device: {cp.cuda.Device().name})")
    logger.info(f"CuPy GPU acceleration: AVAILABLE")
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    logger.info("CuPy GPU acceleration: NOT AVAILABLE (using CPU)")

def get_array_module(arr):
    """Get the array module (numpy or cupy) for the given array"""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    return np


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CalibrationConfig:
    """Configuration for geometry calibration"""
    
    # Dataset directory structure
    # dataset_dir/
    #   slices/           <- input projections
    #     metadata.json   <- geometry and acquisition parameters
    #     0001.raw, 0002.raw, ...
    #   geometry_calib/   <- output calibration results
    
    dataset_dir: str = "CBCT_3D_SheppLogan"
    
    # Derived paths (set automatically)
    input_path: str = ""
    output_path: str = ""
    metadata_path: str = ""
    
    # Dataset parameters (loaded from metadata.json)
    num_projections: int = 720
    angle_step: float = 0.5
    start_angle: float = 0.0
    scan_angle: float = 360.0
    acquisition_direction: str = "CCW"
    
    # Detector parameters
    detector_size: Tuple[int, int] = (3072, 3072)  # (rows, cols)
    pixel_size_mm: Tuple[float, float] = (0.139, 0.139)  # (v, u)
    
    # Nominal geometry (initial estimates)
    source_object_dist: float = 81.454  # mm
    source_detector_dist: float = 814.554  # mm
    detector_offset_u: float = 0.0  # pixels
    detector_offset_v: float = 0.0  # pixels
    detector_tilt: float = 0.0  # degrees
    detector_skew: float = 0.0  # degrees
    angular_offset: float = 0.0  # degrees
    
    # RAW file parameters
    file_format: str = "raw"
    bit_depth: str = "uint16"
    endianness: str = "little"
    header_bytes: int = 0
    raw_resolution: Tuple[int, int] = (3072, 3072)
    
    # Calibration search ranges
    cor_u_range: Tuple[float, float] = (-100.0, 100.0)
    cor_v_range: Tuple[float, float] = (-50.0, 50.0)
    tilt_range: Tuple[float, float] = (-5.0, 5.0)
    skew_range: Tuple[float, float] = (-2.0, 2.0)
    sod_range: Tuple[float, float] = (-5.0, 5.0)  # Relative error in mm
    angular_range: Tuple[float, float] = (-2.0, 2.0)
    
    # Optimization parameters
    coarse_steps: int = 20
    fine_steps: int = 50
    refinement_iterations: int = 3
    
    # GPU acceleration
    use_gpu: bool = True  # Use GPU if available
    calibration_downsample: int = 1  # Downsample projections for faster calibration
    
    # Projection pairs to use (angle indices) - will be recalculated based on num_projections
    calibration_pairs: List[Tuple[int, int]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize derived paths"""
        self._update_paths()
    
    def _update_paths(self):
        """Update input/output paths based on dataset_dir"""
        self.input_path = os.path.join(self.dataset_dir, "slices")
        self.output_path = os.path.join(self.dataset_dir, "geometry_calib")
        self.metadata_path = os.path.join(self.input_path, "metadata.json")
    
    def _calculate_calibration_pairs(self):
        """Calculate projection pairs for 0°/180°, 45°/225°, etc."""
        pairs = []
        half = self.num_projections // 2
        quarter = self.num_projections // 4
        
        # 0° / 180°
        pairs.append((0, half))
        
        # 45° / 225°
        if quarter > 0:
            pairs.append((quarter // 2, half + quarter // 2))
        
        # 90° / 270°
        if quarter > 0:
            pairs.append((quarter, half + quarter))
        
        # 135° / 315°
        if quarter > 0:
            pairs.append((quarter + quarter // 2, half + quarter + quarter // 2))
        
        self.calibration_pairs = pairs
        logger.info(f"Calibration pairs (indices): {pairs}")
    
    @classmethod
    def from_dataset(cls, dataset_dir: str) -> "CalibrationConfig":
        """
        Create configuration from dataset directory
        Reads metadata.json from dataset_dir/slices/metadata.json
        """
        config = cls(dataset_dir=dataset_dir)
        config._update_paths()
        
        if os.path.exists(config.metadata_path):
            config._load_metadata()
            logger.info(f"Loaded metadata from {config.metadata_path}")
        else:
            logger.warning(f"No metadata.json found at {config.metadata_path}, using defaults")
        
        # Auto-detect file format if not specified
        config._auto_detect_format()
        
        config._calculate_calibration_pairs()
        return config
    
    def _auto_detect_format(self):
        """Auto-detect file format from files in input directory"""
        if not os.path.exists(self.input_path):
            return
        
        # Check what files exist
        import glob
        for ext, fmt in [('.tif', 'tiff'), ('.tiff', 'tiff'), ('.raw', 'raw'), 
                         ('.png', 'png'), ('.jpg', 'jpg'), ('.jpeg', 'jpeg')]:
            pattern = os.path.join(self.input_path, f"*{ext}")
            if glob.glob(pattern):
                if self.file_format == "raw" and fmt != "raw":
                    # Override default if we find actual files
                    self.file_format = fmt
                    logger.info(f"Auto-detected file format: {fmt}")
                break
    
    def _load_metadata(self):
        """Load parameters from metadata.json (supports new schema)"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # ============================================================
        # ACQUISITION PARAMETERS
        # ============================================================
        # New schema
        if "acquisition_num_projections" in meta:
            self.num_projections = int(meta["acquisition_num_projections"])
        elif "num_projections" in meta:
            self.num_projections = int(meta["num_projections"])
        
        if "acquisition_angle_step_deg" in meta:
            self.angle_step = float(meta["acquisition_angle_step_deg"])
        elif "angle_step" in meta:
            self.angle_step = float(meta["angle_step"])
        
        if "acquisition_start_angle_deg" in meta:
            self.start_angle = float(meta["acquisition_start_angle_deg"])
        elif "start_angle" in meta:
            self.start_angle = float(meta["start_angle"])
        
        if "acquisition_scan_range_deg" in meta:
            self.scan_angle = float(meta["acquisition_scan_range_deg"])
        elif "scan_angle_degrees" in meta:
            self.scan_angle = float(meta["scan_angle_degrees"])
        elif "scan_angle" in meta:
            self.scan_angle = float(meta["scan_angle"])
        
        if "acquisition_rotation_direction" in meta:
            self.acquisition_direction = str(meta["acquisition_rotation_direction"])
        elif "acquisition_direction" in meta:
            self.acquisition_direction = str(meta["acquisition_direction"])
        
        # ============================================================
        # DETECTOR PARAMETERS
        # ============================================================
        # New schema: separate rows/cols
        if "detector_rows_px" in meta and "detector_cols_px" in meta:
            self.detector_size = (int(meta["detector_rows_px"]), int(meta["detector_cols_px"]))
        elif "detector_pixels" in meta:
            self.detector_size = tuple(int(x) for x in meta["detector_pixels"])
        
        # New schema: separate u/v pixel pitch
        if "detector_pixel_pitch_u_mm" in meta and "detector_pixel_pitch_v_mm" in meta:
            self.pixel_size_mm = (float(meta["detector_pixel_pitch_v_mm"]), 
                                  float(meta["detector_pixel_pitch_u_mm"]))
        elif "pixel_size_mm" in meta:
            self.pixel_size_mm = tuple(float(x) for x in meta["pixel_size_mm"])
        
        # ============================================================
        # GEOMETRY PARAMETERS
        # ============================================================
        # New schema
        if "source_to_origin_dist_mm" in meta:
            self.source_object_dist = float(meta["source_to_origin_dist_mm"])
        elif "source_origin_dist" in meta:
            self.source_object_dist = float(meta["source_origin_dist"])
        
        if "source_to_detector_dist_mm" in meta:
            self.source_detector_dist = float(meta["source_to_detector_dist_mm"])
        elif "source_detector_dist" in meta:
            self.source_detector_dist = float(meta["source_detector_dist"])
        
        # Detector offset (old schema)
        if "detector_offset" in meta:
            offsets = meta["detector_offset"]
            self.detector_offset_u = float(offsets[0])
            self.detector_offset_v = float(offsets[1])
        if "detector_offset_u" in meta:
            self.detector_offset_u = float(meta["detector_offset_u"])
        if "detector_offset_v" in meta:
            self.detector_offset_v = float(meta["detector_offset_v"])
        
        # ============================================================
        # CALIBRATION PARAMETERS (from previous calibration)
        # ============================================================
        if "calibration_offset_u_px" in meta:
            self.detector_offset_u = float(meta["calibration_offset_u_px"])
        if "calibration_offset_v_px" in meta:
            self.detector_offset_v = float(meta["calibration_offset_v_px"])
        if "calibration_tilt_deg" in meta:
            self.detector_tilt = float(meta["calibration_tilt_deg"])
        if "calibration_skew_deg" in meta:
            self.detector_skew = float(meta["calibration_skew_deg"])
        if "calibration_angular_offset_deg" in meta:
            self.angular_offset = float(meta["calibration_angular_offset_deg"])
        
        # ============================================================
        # FILE FORMAT PARAMETERS
        # ============================================================
        if "file_format" in meta:
            self.file_format = str(meta["file_format"]).lower()
        
        # RAW file parameters (new schema)
        if "raw_dtype" in meta:
            self.bit_depth = str(meta["raw_dtype"])
        elif "projection_dtype" in meta:
            self.bit_depth = str(meta["projection_dtype"])
        elif "raw_bit_depth" in meta:
            self.bit_depth = str(meta["raw_bit_depth"])
        
        if "raw_endian" in meta:
            self.endianness = str(meta["raw_endian"]).lower()
        elif "raw_endianness" in meta:
            self.endianness = str(meta["raw_endianness"]).lower()
        
        if "raw_header_bytes" in meta:
            self.header_bytes = int(meta["raw_header_bytes"])
        elif "raw_header_size" in meta:
            hs = meta["raw_header_size"]
            if isinstance(hs, str):
                self.header_bytes = int(hs.split()[0])
            else:
                self.header_bytes = int(hs)
        
        # RAW resolution (new schema)
        if "raw_rows_px" in meta and "raw_cols_px" in meta:
            self.raw_resolution = (int(meta["raw_rows_px"]), int(meta["raw_cols_px"]))
        elif "raw_resolution" in meta:
            res = meta["raw_resolution"]
            if isinstance(res, str):
                parts = res.replace('×', 'x').split('x')
                self.raw_resolution = (int(parts[0].strip()), int(parts[1].strip()))
            else:
                self.raw_resolution = tuple(int(x) for x in res)
        
        # ============================================================
        # PREPROCESSING (for reference, not used in calibration)
        # ============================================================
        if "preprocessing_downsample_factor" in meta:
            ds = int(meta["preprocessing_downsample_factor"])
            if ds > 1:
                logger.info(f"Note: Dataset has preprocessing downsample factor {ds}")
        
        # Log summary
        magnification = self.source_detector_dist / self.source_object_dist
        logger.info(f"Loaded: {self.num_projections} projections, "
                   f"detector {self.detector_size[0]}×{self.detector_size[1]}, "
                   f"SOD={self.source_object_dist}mm, SDD={self.source_detector_dist}mm, "
                   f"mag={magnification:.2f}×")
    
    @classmethod
    def from_json(cls, path: str) -> "CalibrationConfig":
        """Load configuration from JSON file (full config, not metadata)"""
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        config._update_paths()
        config._calculate_calibration_pairs()
        return config
    
    def to_json(self, path: str):
        """Save configuration to JSON file"""
        # Convert to dict, handling tuples
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, tuple):
                data[key] = list(value)
            else:
                data[key] = value
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


@dataclass
class CalibrationResult:
    """Results from geometry calibration"""
    
    # Estimated parameters
    cor_offset_u: float = 0.0
    cor_offset_v: float = 0.0
    detector_tilt: float = 0.0
    detector_skew: float = 0.0
    sod_correction: float = 0.0
    angular_offset: float = 0.0
    
    # Quality metrics
    final_cost: float = float('inf')
    convergence_history: List[float] = field(default_factory=list)
    parameter_history: List[Dict[str, float]] = field(default_factory=list)
    
    # Diagnostics
    residual_maps: Dict[str, np.ndarray] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large arrays)"""
        return {
            "cor_offset_u": self.cor_offset_u,
            "cor_offset_v": self.cor_offset_v,
            "detector_tilt": self.detector_tilt,
            "detector_skew": self.detector_skew,
            "sod_correction": self.sod_correction,
            "angular_offset": self.angular_offset,
            "final_cost": self.final_cost,
            "confidence_scores": self.confidence_scores,
        }
    
    def save(self, path: str):
        """Save results to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class CostMetric(Enum):
    """Cost function options for optimization"""
    CROSS_CORRELATION = "cross_correlation"
    MEAN_SQUARED_ERROR = "mse"
    NORMALIZED_MUTUAL_INFO = "nmi"
    GRADIENT_CORRELATION = "gradient"
    ENTROPY = "entropy"


# =============================================================================
# DATA LOADING
# =============================================================================

class ProjectionLoader:
    """Load and cache projection data with optional GPU acceleration"""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self._cache: Dict[int, np.ndarray] = {}
        self._gpu_cache: Dict[int, Any] = {}  # CuPy arrays
        self._file_list: Optional[List[str]] = None
        self.use_gpu = config.use_gpu and GPU_AVAILABLE
        
        if self.use_gpu:
            logger.info("ProjectionLoader: GPU mode enabled")
        else:
            logger.info("ProjectionLoader: CPU mode")
    
    def _discover_files(self) -> List[str]:
        """Discover projection files in input directory"""
        if self._file_list is not None:
            return self._file_list
        
        import glob
        
        # Determine file extension based on format
        fmt = self.config.file_format.lower()
        if fmt == "raw":
            pattern = os.path.join(self.config.input_path, "*.raw")
        elif fmt in ("tiff", "tif"):
            pattern = os.path.join(self.config.input_path, "*.tif*")
        elif fmt == "png":
            pattern = os.path.join(self.config.input_path, "*.png")
        elif fmt in ("jpg", "jpeg"):
            pattern = os.path.join(self.config.input_path, "*.jp*g")
        else:
            # Try common formats
            all_files = []
            for ext in ("*.raw", "*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
                all_files.extend(glob.glob(os.path.join(self.config.input_path, ext)))
            self._file_list = sorted(all_files)
            return self._file_list
        
        self._file_list = sorted(glob.glob(pattern))
        
        if len(self._file_list) == 0:
            raise FileNotFoundError(f"No projection files found in {self.config.input_path}")
        
        logger.info(f"Found {len(self._file_list)} projection files")
        return self._file_list
    
    def load_projection(self, index: int, to_gpu: bool = None) -> np.ndarray:
        """
        Load single projection by index (0-based)
        
        Args:
            index: Projection index
            to_gpu: Transfer to GPU (default: use config setting)
        
        Returns:
            Projection array (numpy or cupy depending on settings)
        """
        if to_gpu is None:
            to_gpu = self.use_gpu
        
        # Check GPU cache first
        if to_gpu and index in self._gpu_cache:
            return self._gpu_cache[index]
        
        # Check CPU cache
        if index in self._cache:
            proj = self._cache[index]
            if to_gpu:
                proj_gpu = cp.asarray(proj)
                self._gpu_cache[index] = proj_gpu
                return proj_gpu
            return proj
        
        # Load from disk
        files = self._discover_files()
        
        if index < 0 or index >= len(files):
            raise IndexError(f"Projection index {index} out of range [0, {len(files)-1}]")
        
        path = files[index]
        ext = os.path.splitext(path)[1].lower()
        
        if ext == ".raw":
            projection = self._load_raw(path)
        elif ext in (".tif", ".tiff"):
            projection = self._load_tiff(path)
        elif ext in (".png", ".jpg", ".jpeg"):
            projection = self._load_image(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Apply calibration downsampling
        ds = self.config.calibration_downsample
        if ds > 1:
            projection = projection[::ds, ::ds]
        
        # Cache on CPU
        self._cache[index] = projection
        
        # Transfer to GPU if requested
        if to_gpu:
            proj_gpu = cp.asarray(projection)
            self._gpu_cache[index] = proj_gpu
            return proj_gpu
        
        return projection
    
    def _load_raw(self, path: str) -> np.ndarray:
        """Load RAW binary file"""
        dtype = np.dtype(self.config.bit_depth)
        if self.config.endianness == "little":
            dtype = dtype.newbyteorder('<')
        elif self.config.endianness == "big":
            dtype = dtype.newbyteorder('>')
        
        height, width = self.config.raw_resolution
        
        with open(path, "rb") as f:
            if self.config.header_bytes > 0:
                f.seek(self.config.header_bytes)
            data = np.frombuffer(f.read(), dtype=dtype)
        
        return data.reshape((height, width)).astype(np.float32)
    
    def _load_tiff(self, path: str) -> np.ndarray:
        """Load TIFF file (supports 8/16/32-bit, grayscale/RGB, multi-page)"""
        try:
            import tifffile
            arr = tifffile.imread(path)
            
            # Handle multi-page TIFF (take first page)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            elif arr.ndim == 3 and arr.shape[2] in (3, 4):
                # RGB/RGBA -> convert to grayscale
                arr = np.mean(arr[..., :3], axis=-1)
            elif arr.ndim == 3:
                # Multi-page, take first
                arr = arr[0]
            
            return arr.astype(np.float32)
        except ImportError:
            # Fallback to PIL
            from PIL import Image
            with Image.open(path) as img:
                # Convert to grayscale if needed
                if img.mode in ('RGB', 'RGBA'):
                    img = img.convert('L')
                elif img.mode == 'I;16':
                    # 16-bit grayscale
                    return np.array(img, dtype=np.float32)
                return np.array(img, dtype=np.float32)
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load PNG/JPG image file"""
        from PIL import Image
        with Image.open(path) as img:
            return np.array(img.convert("F"), dtype=np.float32)
    
    def load_pair(self, idx_a: int, idx_b: int, to_gpu: bool = None) -> Tuple[Any, Any]:
        """Load a projection pair"""
        return self.load_projection(idx_a, to_gpu), self.load_projection(idx_b, to_gpu)
    
    def get_num_projections(self) -> int:
        """Get total number of available projections"""
        return len(self._discover_files())
    
    def clear_cache(self):
        """Clear projection cache to free memory"""
        self._cache.clear()
        if self.use_gpu:
            self._gpu_cache.clear()
            cp.get_default_memory_pool().free_all_blocks()


# =============================================================================
# GEOMETRY TRANSFORMS
# =============================================================================

class GeometryTransform:
    """Apply geometric transformations to projections (GPU-accelerated if available)"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
    
    def _get_xp(self, arr):
        """Get array module for given array"""
        if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
            return cp
        return np
    
    def flip_horizontal(self, proj):
        """Flip projection horizontally (for 180° comparison)"""
        xp = self._get_xp(proj)
        return xp.fliplr(proj)
    
    def flip_vertical(self, proj):
        """Flip projection vertically"""
        xp = self._get_xp(proj)
        return xp.flipud(proj)
    
    def shift_subpixel(self, proj, shift_u: float, shift_v: float):
        """
        Apply sub-pixel shift using interpolation
        
        Args:
            proj: Input projection (numpy or cupy array)
            shift_u: Horizontal shift (positive = right)
            shift_v: Vertical shift (positive = down)
        """
        if shift_u == 0.0 and shift_v == 0.0:
            return proj.copy()
        
        xp = self._get_xp(proj)
        
        if xp == cp:
            # GPU path
            return gpu_shift(proj, (shift_v, shift_u), mode='constant', cval=0, order=3)
        else:
            # CPU path
            try:
                from scipy.ndimage import shift
                return shift(proj, (shift_v, shift_u), mode='constant', cval=0, order=3)
            except ImportError:
                # Fallback to integer shift
                return self._integer_shift(proj, shift_u, shift_v)
    
    def _integer_shift(self, proj, shift_u: float, shift_v: float):
        """Fallback integer shift"""
        xp = self._get_xp(proj)
        result = xp.zeros_like(proj)
        shift_u_int = int(round(shift_u))
        shift_v_int = int(round(shift_v))
        
        if shift_u_int >= 0 and shift_v_int >= 0:
            result[shift_v_int:, shift_u_int:] = proj[:-shift_v_int or None, :-shift_u_int or None]
        elif shift_u_int >= 0 and shift_v_int < 0:
            result[:shift_v_int, shift_u_int:] = proj[-shift_v_int:, :-shift_u_int or None]
        elif shift_u_int < 0 and shift_v_int >= 0:
            result[shift_v_int:, :shift_u_int] = proj[:-shift_v_int or None, -shift_u_int:]
        else:
            result[:shift_v_int, :shift_u_int] = proj[-shift_v_int:, -shift_u_int:]
        
        return result
    
    def apply_tilt(self, proj, tilt_deg: float, center: Tuple[float, float] = None):
        """
        Apply in-plane rotation (detector tilt)
        
        Args:
            proj: Input projection
            tilt_deg: Rotation angle in degrees (positive = CCW)
            center: Rotation center (row, col) (default: image center)
        """
        if tilt_deg == 0.0:
            return proj.copy()
        
        xp = self._get_xp(proj)
        
        if xp == cp:
            # GPU path
            return gpu_rotate(proj, tilt_deg, reshape=False, mode='constant', cval=0, order=3)
        else:
            # CPU path
            try:
                from scipy.ndimage import rotate
                return rotate(proj, tilt_deg, reshape=False, mode='constant', cval=0, order=3)
            except ImportError:
                logger.warning("scipy not available, skipping tilt transform")
                return proj.copy()
    
    def apply_scale(self, proj, scale_factor: float, center: Tuple[float, float] = None):
        """
        Apply uniform scaling (for SOD correction)
        
        Args:
            proj: Input projection
            scale_factor: Scale multiplier (>1 = enlarge)
            center: Scale center (row, col) (default: image center)
        """
        if scale_factor == 1.0:
            return proj.copy()
        
        xp = self._get_xp(proj)
        rows, cols = proj.shape
        
        if xp == cp:
            # GPU path
            zoomed = gpu_zoom(proj, scale_factor, mode='constant', cval=0, order=3)
        else:
            # CPU path
            try:
                from scipy.ndimage import zoom
                zoomed = zoom(proj, scale_factor, mode='constant', cval=0, order=3)
            except ImportError:
                logger.warning("scipy not available, skipping scale transform")
                return proj.copy()
        
        # Crop or pad to original size, centered
        new_rows, new_cols = zoomed.shape
        result = xp.zeros((rows, cols), dtype=proj.dtype)
        
        if scale_factor > 1.0:
            # Zoomed image is larger -> crop center
            row_offset = (new_rows - rows) // 2
            col_offset = (new_cols - cols) // 2
            result = zoomed[row_offset:row_offset+rows, col_offset:col_offset+cols].copy()
        else:
            # Zoomed image is smaller -> pad with zeros
            row_offset = (rows - new_rows) // 2
            col_offset = (cols - new_cols) // 2
            result[row_offset:row_offset+new_rows, col_offset:col_offset+new_cols] = zoomed
        
        return result
    
    def apply_skew(self, proj, skew_deg: float):
        """
        Apply skew transform (non-orthogonal detector axes)
        
        Args:
            proj: Input projection
            skew_deg: Skew angle in degrees
        """
        if skew_deg == 0.0:
            return proj.copy()
        
        xp = self._get_xp(proj)
        rows, cols = proj.shape
        center_row = rows / 2
        
        skew_rad = xp.radians(skew_deg) if xp == cp else np.radians(skew_deg)
        shear = float(xp.tan(skew_rad)) if xp == cp else np.tan(skew_rad)
        
        matrix = xp.array([[1, 0], [-shear, 1]], dtype=xp.float32)
        offset = xp.array([shear * center_row, 0], dtype=xp.float32)
        
        if xp == cp:
            return gpu_affine_transform(proj, matrix, offset=offset, mode='constant', cval=0, order=3)
        else:
            try:
                from scipy.ndimage import affine_transform
                return affine_transform(proj, matrix, offset=offset, mode='constant', cval=0, order=3)
            except ImportError:
                logger.warning("scipy not available, skipping skew transform")
                return proj.copy()
    
    def apply_full_transform(self, proj, 
                              shift_u: float = 0.0, shift_v: float = 0.0,
                              tilt: float = 0.0, scale: float = 1.0, 
                              skew: float = 0.0):
        """Apply all geometric transforms in correct order"""
        result = proj.copy()
        
        # Order: skew -> scale -> tilt -> shift
        if skew != 0.0:
            result = self.apply_skew(result, skew)
        if scale != 1.0:
            result = self.apply_scale(result, scale)
        if tilt != 0.0:
            result = self.apply_tilt(result, tilt)
        if shift_u != 0.0 or shift_v != 0.0:
            result = self.shift_subpixel(result, shift_u, shift_v)
        
        return result


# =============================================================================
# COST FUNCTIONS
# =============================================================================

class CostFunctions:
    """Cost functions for measuring projection alignment (GPU-accelerated if available)"""
    
    @staticmethod
    def _get_xp(arr):
        """Get array module for given array"""
        if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
            return cp
        return np
    
    @staticmethod
    def cross_correlation(proj_a, proj_b, region: Tuple[slice, slice] = None) -> float:
        """
        Normalized cross-correlation (higher = better alignment)
        Returns negative for minimization
        """
        xp = CostFunctions._get_xp(proj_a)
        
        if region:
            proj_a = proj_a[region]
            proj_b = proj_b[region]
        
        a_mean = xp.mean(proj_a)
        b_mean = xp.mean(proj_b)
        a_std = xp.std(proj_a) + 1e-10
        b_std = xp.std(proj_b) + 1e-10
        
        a_norm = (proj_a - a_mean) / a_std
        b_norm = (proj_b - b_mean) / b_std
        
        correlation = xp.mean(a_norm * b_norm)
        
        # Convert to Python float if on GPU
        if xp == cp:
            correlation = float(correlation.get())
        
        return -correlation  # Negative for minimization
    
    @staticmethod
    def mean_squared_error(proj_a, proj_b, region: Tuple[slice, slice] = None) -> float:
        """Mean squared error (lower = better alignment)"""
        xp = CostFunctions._get_xp(proj_a)
        
        if region:
            proj_a = proj_a[region]
            proj_b = proj_b[region]
        
        # Normalize intensities
        a_norm = (proj_a - xp.mean(proj_a)) / (xp.std(proj_a) + 1e-10)
        b_norm = (proj_b - xp.mean(proj_b)) / (xp.std(proj_b) + 1e-10)
        
        mse = xp.mean((a_norm - b_norm) ** 2)
        
        if xp == cp:
            mse = float(mse.get())
        
        return mse
    
    @staticmethod
    def gradient_correlation(proj_a, proj_b, region: Tuple[slice, slice] = None) -> float:
        """
        Correlation of image gradients (robust to intensity differences)
        """
        xp = CostFunctions._get_xp(proj_a)
        
        if region:
            proj_a = proj_a[region]
            proj_b = proj_b[region]
        
        # Compute gradients using Sobel-like approach (same output size)
        # Horizontal gradient (along columns)
        grad_a_x = proj_a[:, 2:] - proj_a[:, :-2]
        grad_b_x = proj_b[:, 2:] - proj_b[:, :-2]
        
        # Vertical gradient (along rows)
        grad_a_y = proj_a[2:, :] - proj_a[:-2, :]
        grad_b_y = proj_b[2:, :] - proj_b[:-2, :]
        
        # Trim to common size (remove 1 pixel border)
        grad_a_x = grad_a_x[1:-1, :]
        grad_b_x = grad_b_x[1:-1, :]
        grad_a_y = grad_a_y[:, 1:-1]
        grad_b_y = grad_b_y[:, 1:-1]
        
        # Gradient magnitudes
        mag_a = xp.sqrt(grad_a_x**2 + grad_a_y**2)
        mag_b = xp.sqrt(grad_b_x**2 + grad_b_y**2)
        
        # Normalized correlation
        a_mean = xp.mean(mag_a)
        b_mean = xp.mean(mag_b)
        a_std = xp.std(mag_a) + 1e-10
        b_std = xp.std(mag_b) + 1e-10
        
        a_norm = (mag_a - a_mean) / a_std
        b_norm = (mag_b - b_mean) / b_std
        
        corr = -xp.mean(a_norm * b_norm)
        
        if xp == cp:
            corr = float(corr.get())
        
        return corr
    
    @staticmethod
    def regional_costs(proj_a, proj_b, grid: Tuple[int, int] = (3, 3)):
        """
        Compute cost function for multiple regions
        Returns grid of local costs for spatial analysis
        """
        xp = CostFunctions._get_xp(proj_a)
        rows, cols = proj_a.shape
        grid_rows, grid_cols = grid
        
        costs = np.zeros(grid)  # Always return numpy array
        
        for i in range(grid_rows):
            for j in range(grid_cols):
                row_start = i * rows // grid_rows
                row_end = (i + 1) * rows // grid_rows
                col_start = j * cols // grid_cols
                col_end = (j + 1) * cols // grid_cols
                
                region = (slice(row_start, row_end), slice(col_start, col_end))
                costs[i, j] = CostFunctions.cross_correlation(proj_a, proj_b, region)
        
        return costs


# =============================================================================
# PARAMETER ESTIMATORS
# =============================================================================

class ParameterEstimator:
    """Estimate individual calibration parameters (GPU-accelerated if available)"""
    
    def __init__(self, config: CalibrationConfig, loader: ProjectionLoader):
        self.config = config
        self.loader = loader
        self.use_gpu = config.use_gpu and GPU_AVAILABLE
        self.transform = GeometryTransform(use_gpu=self.use_gpu)
        self.cost = CostFunctions()
    
    def estimate_cor_u(self, search_range: Tuple[float, float] = None,
                       num_steps: int = None,
                       current_params: Dict[str, float] = None) -> Tuple[float, np.ndarray]:
        """
        Estimate horizontal COR offset using multiple projection pairs
        
        Args:
            search_range: (min, max) offset range in pixels
            num_steps: Number of steps in search
            current_params: Current estimates of other parameters
        
        Returns:
            best_offset: Estimated COR offset in pixels
            cost_curve: Cost values for each tested offset
        """
        if search_range is None:
            search_range = self.config.cor_u_range
        if num_steps is None:
            num_steps = self.config.coarse_steps
        if current_params is None:
            current_params = {}
        
        offsets = np.linspace(search_range[0], search_range[1], num_steps)
        costs = np.zeros(len(offsets))
        
        for pair in self.config.calibration_pairs:
            proj_a, proj_b = self.loader.load_pair(pair[0], pair[1])
            proj_b_flip = self.transform.flip_horizontal(proj_b)
            
            for i, offset in enumerate(offsets):
                shift = -offset  # COR offset = required shift
                proj_b_shifted = self.transform.apply_full_transform(
                    proj_b_flip,
                    shift_u=shift,
                    shift_v=-current_params.get('cor_v', 0),
                    tilt=current_params.get('tilt', 0),
                    scale=1.0 + current_params.get('sod_scale', 0),
                    skew=current_params.get('skew', 0)
                )
                costs[i] += self.cost.gradient_correlation(proj_a, proj_b_shifted)
        
        costs /= len(self.config.calibration_pairs)
        best_idx = np.argmin(costs)
        
        return offsets[best_idx], costs
    
    def estimate_cor_v(self, search_range: Tuple[float, float] = None,
                       num_steps: int = None,
                       current_params: Dict[str, float] = None) -> Tuple[float, np.ndarray]:
        """
        Estimate vertical COR offset using multiple projection pairs
        
        Returns:
            best_offset: Estimated COR offset in pixels
            cost_curve: Cost values for each tested offset
        """
        if search_range is None:
            search_range = self.config.cor_v_range
        if num_steps is None:
            num_steps = self.config.coarse_steps
        if current_params is None:
            current_params = {}
        
        offsets = np.linspace(search_range[0], search_range[1], num_steps)
        costs = np.zeros(len(offsets))
        
        for pair in self.config.calibration_pairs:
            proj_a, proj_b = self.loader.load_pair(pair[0], pair[1])
            proj_b_flip = self.transform.flip_horizontal(proj_b)
            
            for i, offset in enumerate(offsets):
                shift = -offset
                proj_b_shifted = self.transform.apply_full_transform(
                    proj_b_flip,
                    shift_u=-current_params.get('cor_u', 0),
                    shift_v=shift,
                    tilt=current_params.get('tilt', 0),
                    scale=1.0 + current_params.get('sod_scale', 0),
                    skew=current_params.get('skew', 0)
                )
                costs[i] += self.cost.gradient_correlation(proj_a, proj_b_shifted)
        
        costs /= len(self.config.calibration_pairs)
        best_idx = np.argmin(costs)
        
        return offsets[best_idx], costs
    
    def estimate_tilt(self, search_range: Tuple[float, float] = None,
                      num_steps: int = None,
                      current_params: Dict[str, float] = None) -> Tuple[float, np.ndarray]:
        """
        Estimate detector tilt (in-plane rotation)
        
        Approach: Search for tilt angle that minimizes alignment cost
        """
        if search_range is None:
            search_range = self.config.tilt_range
        if num_steps is None:
            num_steps = self.config.coarse_steps
        if current_params is None:
            current_params = {}
        
        tilts = np.linspace(search_range[0], search_range[1], num_steps)
        costs = np.zeros(len(tilts))
        
        for pair in self.config.calibration_pairs:
            proj_a, proj_b = self.loader.load_pair(pair[0], pair[1])
            proj_b_flip = self.transform.flip_horizontal(proj_b)
            
            for i, tilt in enumerate(tilts):
                proj_b_transformed = self.transform.apply_full_transform(
                    proj_b_flip,
                    shift_u=-current_params.get('cor_u', 0),
                    shift_v=-current_params.get('cor_v', 0),
                    tilt=tilt,
                    scale=1.0 + current_params.get('sod_scale', 0),
                    skew=current_params.get('skew', 0)
                )
                costs[i] += self.cost.gradient_correlation(proj_a, proj_b_transformed)
        
        costs /= len(self.config.calibration_pairs)
        best_idx = np.argmin(costs)
        
        return tilts[best_idx], costs
    
    def estimate_sod_error(self, search_range: Tuple[float, float] = None,
                           num_steps: int = None,
                           current_params: Dict[str, float] = None) -> Tuple[float, np.ndarray]:
        """
        Estimate SOD error from magnification mismatch
        
        Returns scale correction factor (0 = no correction, 0.01 = 1% scale increase)
        """
        if search_range is None:
            # Convert mm range to relative scale
            # scale_change ≈ -sod_error / SOD
            sod = self.config.source_object_dist
            search_range = (-self.config.sod_range[1] / sod, -self.config.sod_range[0] / sod)
        if num_steps is None:
            num_steps = self.config.coarse_steps
        if current_params is None:
            current_params = {}
        
        scales = np.linspace(search_range[0], search_range[1], num_steps)
        costs = np.zeros(len(scales))
        
        for pair in self.config.calibration_pairs:
            proj_a, proj_b = self.loader.load_pair(pair[0], pair[1])
            proj_b_flip = self.transform.flip_horizontal(proj_b)
            
            for i, scale_offset in enumerate(scales):
                proj_b_transformed = self.transform.apply_full_transform(
                    proj_b_flip,
                    shift_u=-current_params.get('cor_u', 0),
                    shift_v=-current_params.get('cor_v', 0),
                    tilt=current_params.get('tilt', 0),
                    scale=1.0 + scale_offset,
                    skew=current_params.get('skew', 0)
                )
                costs[i] += self.cost.gradient_correlation(proj_a, proj_b_transformed)
        
        costs /= len(self.config.calibration_pairs)
        best_idx = np.argmin(costs)
        
        # Convert scale back to SOD error in mm
        best_scale = scales[best_idx]
        sod_error = -best_scale * self.config.source_object_dist
        
        return best_scale, costs
    
    def estimate_skew(self, search_range: Tuple[float, float] = None,
                      num_steps: int = None,
                      current_params: Dict[str, float] = None) -> Tuple[float, np.ndarray]:
        """
        Estimate detector skew (non-orthogonal axes)
        """
        if search_range is None:
            search_range = self.config.skew_range
        if num_steps is None:
            num_steps = self.config.coarse_steps
        if current_params is None:
            current_params = {}
        
        skews = np.linspace(search_range[0], search_range[1], num_steps)
        costs = np.zeros(len(skews))
        
        for pair in self.config.calibration_pairs:
            proj_a, proj_b = self.loader.load_pair(pair[0], pair[1])
            proj_b_flip = self.transform.flip_horizontal(proj_b)
            
            for i, skew in enumerate(skews):
                proj_b_transformed = self.transform.apply_full_transform(
                    proj_b_flip,
                    shift_u=-current_params.get('cor_u', 0),
                    shift_v=-current_params.get('cor_v', 0),
                    tilt=current_params.get('tilt', 0),
                    scale=1.0 + current_params.get('sod_scale', 0),
                    skew=skew
                )
                costs[i] += self.cost.gradient_correlation(proj_a, proj_b_transformed)
        
        costs /= len(self.config.calibration_pairs)
        best_idx = np.argmin(costs)
        
        return skews[best_idx], costs
    
    def estimate_angular_offset(self, search_range: Tuple[float, float] = None,
                                 num_steps: int = None) -> Tuple[float, np.ndarray]:
        """
        Estimate angular indexing offset by testing different projection pair alignments
        
        Approach: Shift the pairing indices and find best alignment
        """
        if search_range is None:
            search_range = self.config.angular_range
        if num_steps is None:
            num_steps = self.config.coarse_steps
        
        # Convert angular offset to index offset
        angle_to_idx = 1.0 / self.config.angle_step
        idx_range = (int(search_range[0] * angle_to_idx), int(search_range[1] * angle_to_idx))
        
        offsets = np.arange(idx_range[0], idx_range[1] + 1)
        costs = np.zeros(len(offsets))
        
        half = self.config.num_projections // 2
        
        for i, idx_offset in enumerate(offsets):
            pair_cost = 0.0
            valid_pairs = 0
            
            for pair in self.config.calibration_pairs:
                idx_a = pair[0]
                idx_b = (pair[1] + idx_offset) % self.config.num_projections
                
                try:
                    proj_a, proj_b = self.loader.load_pair(idx_a, idx_b)
                    proj_b_flip = self.transform.flip_horizontal(proj_b)
                    pair_cost += self.cost.gradient_correlation(proj_a, proj_b_flip)
                    valid_pairs += 1
                except (IndexError, FileNotFoundError):
                    continue
            
            if valid_pairs > 0:
                costs[i] = pair_cost / valid_pairs
            else:
                costs[i] = float('inf')
        
        best_idx = np.argmin(costs)
        best_angular_offset = offsets[best_idx] * self.config.angle_step
        
        return best_angular_offset, costs
    
    def estimate_local_cor(self, region: Tuple[slice, slice],
                           search_range: Tuple[float, float] = None,
                           num_steps: int = 50) -> float:
        """
        Estimate COR for a specific region of the detector
        Used for detecting tilt (top vs bottom COR difference)
        """
        if search_range is None:
            search_range = self.config.cor_u_range
        
        offsets = np.linspace(search_range[0], search_range[1], num_steps)
        costs = np.zeros(len(offsets))
        
        for pair in self.config.calibration_pairs:
            proj_a, proj_b = self.loader.load_pair(pair[0], pair[1])
            proj_b_flip = self.transform.flip_horizontal(proj_b)
            
            # Extract regions
            proj_a_region = proj_a[region]
            proj_b_region = proj_b_flip[region]
            
            for i, offset in enumerate(offsets):
                shift = -offset
                proj_b_shifted = self.transform.shift_subpixel(proj_b_region, shift, 0)
                costs[i] += self.cost.gradient_correlation(proj_a_region, proj_b_shifted)
        
        costs /= len(self.config.calibration_pairs)
        best_idx = np.argmin(costs)
        
        return offsets[best_idx]
    
    def analyze_spatial_variation(self, grid: Tuple[int, int] = (5, 5)) -> Dict[str, np.ndarray]:
        """
        Analyze how alignment varies across detector
        
        Returns maps showing local COR offset at different positions
        Useful for diagnosing tilt, skew, and SOD errors
        """
        results = {}
        
        # Load first pair to get dimensions
        proj_a, proj_b = self.loader.load_pair(
            self.config.calibration_pairs[0][0],
            self.config.calibration_pairs[0][1]
        )
        rows, cols = proj_a.shape
        
        # Compute local COR for each grid cell
        local_cor_map = np.zeros(grid)
        grid_rows, grid_cols = grid
        
        logger.info(f"Computing local COR for {grid_rows}x{grid_cols} grid...")
        
        for i in range(grid_rows):
            for j in range(grid_cols):
                row_start = i * rows // grid_rows
                row_end = (i + 1) * rows // grid_rows
                col_start = j * cols // grid_cols
                col_end = (j + 1) * cols // grid_cols
                
                region = (slice(row_start, row_end), slice(col_start, col_end))
                local_cor_map[i, j] = self.estimate_local_cor(region, num_steps=30)
        
        results["local_cor_map"] = local_cor_map
        
        # Analyze patterns
        # Vertical gradient indicates tilt
        vertical_gradient = np.mean(np.diff(local_cor_map, axis=0))
        results["vertical_gradient"] = vertical_gradient
        
        # Horizontal gradient indicates skew
        horizontal_gradient = np.mean(np.diff(local_cor_map, axis=1))
        results["horizontal_gradient"] = horizontal_gradient
        
        # Radial pattern indicates SOD error
        center_cor = local_cor_map[grid_rows//2, grid_cols//2]
        corner_cors = [
            local_cor_map[0, 0], local_cor_map[0, -1],
            local_cor_map[-1, 0], local_cor_map[-1, -1]
        ]
        radial_difference = np.mean(corner_cors) - center_cor
        results["radial_difference"] = radial_difference
        
        logger.info(f"  Vertical gradient (tilt indicator): {vertical_gradient:.3f} px/cell")
        logger.info(f"  Horizontal gradient (skew indicator): {horizontal_gradient:.3f} px/cell")
        logger.info(f"  Radial difference (SOD indicator): {radial_difference:.3f} px")
        
        return results


# =============================================================================
# MULTI-PARAMETER OPTIMIZER
# =============================================================================

class GeometryOptimizer:
    """Joint optimization of all geometry parameters (GPU-accelerated if available)"""
    
    def __init__(self, config: CalibrationConfig, loader: ProjectionLoader):
        self.config = config
        self.loader = loader
        self.use_gpu = config.use_gpu and GPU_AVAILABLE
        self.transform = GeometryTransform(use_gpu=self.use_gpu)
        self.cost_fn = CostFunctions()
    
    def _compute_total_cost(self, params: Dict[str, float]) -> float:
        """
        Compute total cost for given parameter set
        
        Args:
            params: Dictionary with keys: cor_u, cor_v, tilt, skew, sod_scale
        """
        total_cost = 0.0
        
        for pair in self.config.calibration_pairs:
            proj_a, proj_b = self.loader.load_pair(pair[0], pair[1])
            proj_b_flip = self.transform.flip_horizontal(proj_b)
            
            # Apply transforms based on parameters
            proj_b_transformed = self.transform.apply_full_transform(
                proj_b_flip,
                shift_u=-params.get('cor_u', 0),
                shift_v=-params.get('cor_v', 0),
                tilt=params.get('tilt', 0),
                scale=1.0 + params.get('sod_scale', 0),
                skew=params.get('skew', 0)
            )
            
            total_cost += self.cost_fn.gradient_correlation(proj_a, proj_b_transformed)
        
        return total_cost / len(self.config.calibration_pairs)
    
    def optimize_sequential(self) -> CalibrationResult:
        """
        Sequential parameter optimization (one parameter at a time)
        
        Faster but may not find global optimum for coupled parameters
        """
        result = CalibrationResult()
        estimator = ParameterEstimator(self.config, self.loader)
        
        logger.info("Starting sequential optimization...")
        
        # Current best parameters
        current_params = {
            'cor_u': 0.0,
            'cor_v': 0.0,
            'tilt': 0.0,
            'sod_scale': 0.0,
            'skew': 0.0
        }
        
        # Iterate to allow parameters to refine each other
        for iteration in range(self.config.refinement_iterations):
            logger.info(f"\nIteration {iteration + 1}/{self.config.refinement_iterations}")
            
            # Estimate COR_u
            cor_u, costs_u = estimator.estimate_cor_u(current_params=current_params)
            current_params['cor_u'] = cor_u
            result.cor_offset_u = cor_u
            logger.info(f"  COR_u: {cor_u:.3f} px")
            
            # Estimate COR_v
            cor_v, costs_v = estimator.estimate_cor_v(current_params=current_params)
            current_params['cor_v'] = cor_v
            result.cor_offset_v = cor_v
            logger.info(f"  COR_v: {cor_v:.3f} px")
            
            # Estimate tilt
            tilt, costs_tilt = estimator.estimate_tilt(current_params=current_params)
            current_params['tilt'] = tilt
            result.detector_tilt = tilt
            logger.info(f"  Tilt:  {tilt:.3f} deg")
            
            # Estimate skew
            skew, costs_skew = estimator.estimate_skew(current_params=current_params)
            current_params['skew'] = skew
            result.detector_skew = skew
            logger.info(f"  Skew:  {skew:.3f} deg")
            
            # Estimate SOD scale
            sod_scale, costs_sod = estimator.estimate_sod_error(current_params=current_params)
            current_params['sod_scale'] = sod_scale
            result.sod_correction = -sod_scale * self.config.source_object_dist
            logger.info(f"  SOD:   {result.sod_correction:.3f} mm")
            
            # Compute current cost
            cost = self._compute_total_cost(current_params)
            result.convergence_history.append(cost)
            result.parameter_history.append(current_params.copy())
            logger.info(f"  Cost:  {cost:.6f}")
        
        result.final_cost = result.convergence_history[-1]
        return result
    
    def optimize_joint(self, method: str = "powell") -> CalibrationResult:
        """
        Joint optimization of all parameters simultaneously
        
        Uses scipy.optimize for multi-dimensional minimization
        """
        from scipy.optimize import minimize
        
        result = CalibrationResult()
        
        logger.info(f"Starting joint optimization (method: {method})...")
        
        # Parameter order: cor_u, cor_v, tilt, skew, sod_scale
        param_names = ['cor_u', 'cor_v', 'tilt', 'skew', 'sod_scale']
        
        # Bounds for each parameter
        bounds = [
            self.config.cor_u_range,
            self.config.cor_v_range,
            self.config.tilt_range,
            self.config.skew_range,
            (-0.1, 0.1)  # SOD scale range
        ]
        
        def objective(x):
            params = {name: val for name, val in zip(param_names, x)}
            cost = self._compute_total_cost(params)
            result.convergence_history.append(cost)
            return cost
        
        # Initial guess (zeros or from metadata)
        x0 = np.array([
            self.config.detector_offset_u,
            self.config.detector_offset_v,
            self.config.detector_tilt,
            self.config.detector_skew,
            0.0
        ])
        
        logger.info(f"  Initial guess: {x0}")
        
        # Run optimization
        if method in ('powell', 'nelder-mead', 'cobyla'):
            opt_result = minimize(objective, x0, method=method,
                                  options={'maxiter': 200, 'disp': True})
        else:
            # Methods that support bounds
            opt_result = minimize(objective, x0, method=method, bounds=bounds,
                                  options={'maxiter': 200, 'disp': True})
        
        # Extract results
        result.cor_offset_u = opt_result.x[0]
        result.cor_offset_v = opt_result.x[1]
        result.detector_tilt = opt_result.x[2]
        result.detector_skew = opt_result.x[3]
        result.sod_correction = -opt_result.x[4] * self.config.source_object_dist
        result.final_cost = opt_result.fun
        
        logger.info(f"\nOptimization {'converged' if opt_result.success else 'failed'}")
        logger.info(f"  Final parameters: {opt_result.x}")
        logger.info(f"  Final cost: {opt_result.fun:.6f}")
        
        return result
    
    def optimize_coarse_to_fine(self) -> CalibrationResult:
        """
        Coarse-to-fine optimization strategy
        
        1. Coarse grid search for approximate parameters
        2. Fine optimization around best coarse result
        3. Local refinement with joint optimization
        """
        result = CalibrationResult()
        estimator = ParameterEstimator(self.config, self.loader)
        
        logger.info("Phase 1: Coarse grid search...")
        
        # Coarse estimation of each parameter
        coarse_params = {}
        
        # COR_u - most important, do first
        cor_u, _ = estimator.estimate_cor_u(num_steps=self.config.coarse_steps)
        coarse_params['cor_u'] = cor_u
        logger.info(f"  Coarse COR_u: {cor_u:.2f} px")
        
        # COR_v
        cor_v, _ = estimator.estimate_cor_v(num_steps=self.config.coarse_steps,
                                            current_params=coarse_params)
        coarse_params['cor_v'] = cor_v
        logger.info(f"  Coarse COR_v: {cor_v:.2f} px")
        
        # Tilt
        tilt, _ = estimator.estimate_tilt(num_steps=self.config.coarse_steps,
                                          current_params=coarse_params)
        coarse_params['tilt'] = tilt
        logger.info(f"  Coarse tilt: {tilt:.2f} deg")
        
        logger.info("\nPhase 2: Fine grid search...")
        
        # Fine search around coarse result
        fine_range_u = (cor_u - 10, cor_u + 10)
        fine_range_v = (cor_v - 5, cor_v + 5)
        fine_range_tilt = (tilt - 1, tilt + 1)
        
        cor_u_fine, _ = estimator.estimate_cor_u(
            search_range=fine_range_u,
            num_steps=self.config.fine_steps,
            current_params=coarse_params
        )
        coarse_params['cor_u'] = cor_u_fine
        logger.info(f"  Fine COR_u: {cor_u_fine:.3f} px")
        
        cor_v_fine, _ = estimator.estimate_cor_v(
            search_range=fine_range_v,
            num_steps=self.config.fine_steps,
            current_params=coarse_params
        )
        coarse_params['cor_v'] = cor_v_fine
        logger.info(f"  Fine COR_v: {cor_v_fine:.3f} px")
        
        tilt_fine, _ = estimator.estimate_tilt(
            search_range=fine_range_tilt,
            num_steps=self.config.fine_steps,
            current_params=coarse_params
        )
        coarse_params['tilt'] = tilt_fine
        logger.info(f"  Fine tilt: {tilt_fine:.3f} deg")
        
        logger.info("\nPhase 3: Local joint optimization...")
        
        # Set initial guess from fine search
        self.config.detector_offset_u = cor_u_fine
        self.config.detector_offset_v = cor_v_fine
        self.config.detector_tilt = tilt_fine
        
        # Run local optimization
        try:
            result = self.optimize_joint(method='powell')
        except ImportError:
            logger.warning("scipy not available, using fine grid search result")
            result.cor_offset_u = cor_u_fine
            result.cor_offset_v = cor_v_fine
            result.detector_tilt = tilt_fine
            result.final_cost = self._compute_total_cost(coarse_params)
        
        return result


# =============================================================================
# RECONSTRUCTION-BASED VALIDATION
# =============================================================================

class ReconstructionValidator:
    """Validate calibration using reconstruction quality metrics"""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
    
    def compute_sharpness(self, volume: np.ndarray) -> float:
        """
        Compute sharpness metric (higher = sharper)
        Based on gradient magnitude
        """
        grad_x = np.diff(volume, axis=0)
        grad_y = np.diff(volume, axis=1)
        grad_z = np.diff(volume, axis=2)
        
        # Use variance of gradient as sharpness metric
        sharpness = (np.var(grad_x) + np.var(grad_y) + np.var(grad_z)) / 3
        return sharpness
    
    def compute_entropy(self, volume: np.ndarray, bins: int = 256) -> float:
        """
        Compute image entropy (lower = more structured)
        Well-reconstructed images have lower entropy
        """
        hist, _ = np.histogram(volume.flatten(), bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    def compute_tv(self, volume: np.ndarray) -> float:
        """
        Compute total variation (lower = smoother)
        """
        grad_x = np.abs(np.diff(volume, axis=0))
        grad_y = np.abs(np.diff(volume, axis=1))
        grad_z = np.abs(np.diff(volume, axis=2))
        
        tv = np.mean(grad_x) + np.mean(grad_y) + np.mean(grad_z)
        return tv
    
    def validate_calibration(self, result: CalibrationResult,
                             reconstructor: Callable) -> Dict[str, float]:
        """
        Validate calibration by performing reconstruction and measuring quality
        
        Args:
            result: Calibration result to validate
            reconstructor: Function that takes geometry params and returns volume
        """
        # TODO: Implement reconstruction-based validation
        # 1. Reconstruct with calibrated geometry
        # 2. Compute quality metrics
        # 3. Compare to reconstruction without calibration
        
        raise NotImplementedError("Implement reconstruction validation")


# =============================================================================
# VISUALIZATION
# =============================================================================

class CalibrationVisualizer:
    """Visualization tools for calibration results"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def plot_cost_curve(self, parameter_name: str, values: np.ndarray, 
                        costs: np.ndarray, best_value: float, filename: str = None):
        """Plot cost function vs parameter value"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(values, costs, 'b-', linewidth=2)
            ax.axvline(x=best_value, color='r', linestyle='--', linewidth=2,
                       label=f'Best: {best_value:.3f}')
            ax.set_xlabel(parameter_name, fontsize=12)
            ax.set_ylabel('Cost (negative correlation)', fontsize=12)
            ax.set_title(f'Cost vs {parameter_name}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if filename is None:
                filename = f"cost_{parameter_name.lower().replace(' ', '_')}.png"
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=150)
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
    
    def plot_convergence(self, history: List[float], filename: str = "convergence.png"):
        """Plot optimization convergence history"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(len(history)), history, 'b-o', linewidth=2, markersize=8)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Cost', fontsize=12)
            ax.set_title('Optimization Convergence', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=150)
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
    
    def plot_projection_comparison(self, proj_a, proj_b,
                                    proj_b_corrected, title: str,
                                    filename: str = "projection_comparison.png"):
        """Plot before/after comparison of projection alignment"""
        try:
            import matplotlib.pyplot as plt
            
            # Convert CuPy arrays to NumPy if necessary
            if GPU_AVAILABLE:
                if isinstance(proj_a, cp.ndarray):
                    proj_a = cp.asnumpy(proj_a)
                if isinstance(proj_b, cp.ndarray):
                    proj_b = cp.asnumpy(proj_b)
                if isinstance(proj_b_corrected, cp.ndarray):
                    proj_b_corrected = cp.asnumpy(proj_b_corrected)
            
            # Normalize for display
            def normalize(img):
                vmin, vmax = np.percentile(img, [1, 99])
                return np.clip((img - vmin) / (vmax - vmin + 1e-10), 0, 1)
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Top row: projections
            axes[0, 0].imshow(normalize(proj_a), cmap='gray')
            axes[0, 0].set_title('Projection A (0°)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(normalize(proj_b), cmap='gray')
            axes[0, 1].set_title('Projection B (180° flipped)')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(normalize(proj_b_corrected), cmap='gray')
            axes[0, 2].set_title('Projection B (corrected)')
            axes[0, 2].axis('off')
            
            # Bottom row: differences
            diff_before = np.abs(proj_a - proj_b)
            diff_after = np.abs(proj_a - proj_b_corrected)
            
            im1 = axes[1, 0].imshow(diff_before, cmap='hot')
            axes[1, 0].set_title(f'Difference Before\nMean: {np.mean(diff_before):.1f}')
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
            
            im2 = axes[1, 1].imshow(diff_after, cmap='hot')
            axes[1, 1].set_title(f'Difference After\nMean: {np.mean(diff_after):.1f}')
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
            
            # Improvement map
            improvement = diff_before - diff_after
            im3 = axes[1, 2].imshow(improvement, cmap='RdYlGn')
            axes[1, 2].set_title('Improvement (green=better)')
            axes[1, 2].axis('off')
            plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
            
            plt.suptitle(title, fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=150)
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
    
    def plot_spatial_variation(self, variation_map: np.ndarray, title: str,
                               filename: str = "spatial_variation.png"):
        """Plot spatial variation of alignment across detector"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(variation_map, cmap='RdBu_r', interpolation='nearest')
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Detector column region')
            ax.set_ylabel('Detector row region')
            
            # Add value annotations
            for i in range(variation_map.shape[0]):
                for j in range(variation_map.shape[1]):
                    ax.text(j, i, f'{variation_map[i, j]:.1f}',
                            ha='center', va='center', fontsize=10)
            
            plt.colorbar(im, ax=ax, label='Local COR offset (pixels)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=150)
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
    
    def plot_residual_map(self, residual: np.ndarray, title: str,
                          filename: str = "residual_map.png"):
        """Plot residual difference after calibration"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(residual, cmap='hot')
            ax.set_title(title, fontsize=14)
            ax.axis('off')
            plt.colorbar(im, ax=ax, label='Residual')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=150)
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
    
    def generate_report(self, result: CalibrationResult, config: CalibrationConfig = None,
                        spatial_analysis: Dict = None):
        """Generate comprehensive text report and calibration JSON"""
        report_path = os.path.join(self.output_dir, "calibration_report.txt")
        
        # Get current date
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CT GEOMETRY CALIBRATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            if config:
                f.write("DATASET INFORMATION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Dataset: {config.dataset_dir}\n")
                f.write(f"Projections: {config.num_projections}\n")
                f.write(f"Detector size: {config.detector_size[0]} x {config.detector_size[1]}\n")
                f.write(f"Pixel size: {config.pixel_size_mm[0]:.4f} x {config.pixel_size_mm[1]:.4f} mm\n")
                f.write(f"SOD: {config.source_object_dist:.3f} mm\n")
                f.write(f"SDD: {config.source_detector_dist:.3f} mm\n")
                f.write(f"Magnification: {config.source_detector_dist/config.source_object_dist:.2f}x\n\n")
            
            f.write("CALIBRATION RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"COR offset (horizontal): {result.cor_offset_u:.4f} pixels\n")
            f.write(f"COR offset (vertical):   {result.cor_offset_v:.4f} pixels\n")
            f.write(f"Detector tilt:           {result.detector_tilt:.4f} degrees\n")
            f.write(f"Detector skew:           {result.detector_skew:.4f} degrees\n")
            f.write(f"SOD correction:          {result.sod_correction:.4f} mm\n")
            f.write(f"Angular offset:          {result.angular_offset:.4f} degrees\n")
            f.write(f"Final cost:              {result.final_cost:.6f}\n\n")
            
            if config:
                # Convert to physical units
                cor_u_mm = result.cor_offset_u * config.pixel_size_mm[1]
                cor_v_mm = result.cor_offset_v * config.pixel_size_mm[0]
                f.write("PHYSICAL OFFSETS\n")
                f.write("-" * 40 + "\n")
                f.write(f"COR offset (horizontal): {cor_u_mm:.4f} mm\n")
                f.write(f"COR offset (vertical):   {cor_v_mm:.4f} mm\n\n")
            
            if spatial_analysis and 'local_cor_map' in spatial_analysis:
                f.write("SPATIAL ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Vertical gradient (tilt indicator):   {spatial_analysis.get('vertical_gradient', 0):.4f} px/cell\n")
                f.write(f"Horizontal gradient (skew indicator): {spatial_analysis.get('horizontal_gradient', 0):.4f} px/cell\n")
                f.write(f"Radial difference (SOD indicator):    {spatial_analysis.get('radial_difference', 0):.4f} px\n\n")
            
            if result.convergence_history:
                f.write("CONVERGENCE HISTORY\n")
                f.write("-" * 40 + "\n")
                for i, cost in enumerate(result.convergence_history):
                    f.write(f"Iteration {i+1}: {cost:.6f}\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("HOW TO APPLY THESE CORRECTIONS\n")
            f.write("=" * 70 + "\n\n")
            f.write("Add/update these fields in your metadata.json:\n\n")
            f.write('  "_comment_calibration": "========== CALIBRATION PARAMETERS ==========",\n')
            f.write(f'  "calibration_offset_u_px": {result.cor_offset_u:.4f},\n')
            f.write(f'  "calibration_offset_v_px": {result.cor_offset_v:.4f},\n')
            f.write(f'  "calibration_offset_reference": "fullres",\n')
            f.write(f'  "calibration_tilt_deg": {result.detector_tilt:.4f},\n')
            f.write(f'  "calibration_skew_deg": {result.detector_skew:.4f},\n')
            f.write(f'  "calibration_sod_correction_mm": {result.sod_correction:.4f},\n')
            f.write(f'  "calibration_angular_offset_deg": {result.angular_offset:.4f},\n')
            f.write(f'  "calibration_method": "auto",\n')
            f.write(f'  "calibration_final_cost": {result.final_cost:.6f},\n')
            f.write(f'  "calibration_date": "{today}"\n\n')
            
            f.write("=" * 70 + "\n")
        
        logger.info(f"Report saved to {report_path}")
        
        # Also save calibration parameters as JSON for easy copy-paste
        calibration_json_path = os.path.join(self.output_dir, "calibration_params.json")
        calibration_params = {
            "_comment_calibration": "========== CALIBRATION PARAMETERS (from geometry calibration) ==========",
            "calibration_offset_u_px": round(result.cor_offset_u, 4),
            "calibration_offset_v_px": round(result.cor_offset_v, 4),
            "calibration_offset_reference": "fullres",
            "calibration_tilt_deg": round(result.detector_tilt, 4),
            "calibration_skew_deg": round(result.detector_skew, 4),
            "calibration_sod_correction_mm": round(result.sod_correction, 4),
            "calibration_angular_offset_deg": round(result.angular_offset, 4),
            "calibration_method": "auto",
            "calibration_final_cost": round(result.final_cost, 6),
            "calibration_date": today
        }
        
        with open(calibration_json_path, 'w') as f:
            json.dump(calibration_params, f, indent=2)
        
        logger.info(f"Calibration params saved to {calibration_json_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class CalibrationPipeline:
    """Main calibration pipeline orchestrator"""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.loader = ProjectionLoader(config)
        self.optimizer = GeometryOptimizer(config, self.loader)
        self.visualizer = CalibrationVisualizer(config.output_path)
        self.use_gpu = config.use_gpu and GPU_AVAILABLE
        self.transform = GeometryTransform(use_gpu=self.use_gpu)
    
    def run(self, method: str = "sequential") -> CalibrationResult:
        """
        Run complete calibration pipeline
        
        Args:
            method: "sequential", "joint", or "coarse_to_fine"
        """
        logger.info("=" * 60)
        logger.info("CT Geometry Calibration Pipeline")
        logger.info("=" * 60)
        logger.info(f"Dataset: {self.config.dataset_dir}")
        logger.info(f"Method: {method}")
        logger.info(f"Projections: {self.config.num_projections}")
        logger.info(f"Calibration pairs: {self.config.calibration_pairs}")
        
        # Phase 1: Initial diagnostics
        logger.info("\n" + "=" * 60)
        logger.info("Phase 1: Analyzing spatial alignment variation...")
        logger.info("=" * 60)
        
        estimator = ParameterEstimator(self.config, self.loader)
        spatial_analysis = estimator.analyze_spatial_variation(grid=(5, 5))
        
        # Plot spatial variation
        if 'local_cor_map' in spatial_analysis:
            self.visualizer.plot_spatial_variation(
                spatial_analysis['local_cor_map'],
                'Local COR Offset Across Detector',
                'spatial_cor_variation.png'
            )
        
        # Phase 2: Parameter optimization
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: Optimizing geometry parameters...")
        logger.info("=" * 60)
        
        if method == "sequential":
            result = self.optimizer.optimize_sequential()
        elif method == "joint":
            result = self.optimizer.optimize_joint()
        elif method == "coarse_to_fine":
            result = self.optimizer.optimize_coarse_to_fine()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Phase 3: Visualization
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3: Generating visualizations...")
        logger.info("=" * 60)
        
        # Plot convergence
        if result.convergence_history:
            self.visualizer.plot_convergence(result.convergence_history)
        
        # Generate before/after comparison for first projection pair
        if self.config.calibration_pairs:
            pair = self.config.calibration_pairs[0]
            # Load on CPU for visualization
            proj_a, proj_b = self.loader.load_pair(pair[0], pair[1], to_gpu=False)
            proj_b_flip = np.fliplr(proj_b)  # CPU flip
            
            # Apply transform on CPU
            from scipy.ndimage import shift as cpu_shift, rotate as cpu_rotate
            proj_b_corrected = proj_b_flip.copy()
            if result.detector_skew != 0.0:
                from scipy.ndimage import affine_transform
                rows, cols = proj_b_corrected.shape
                skew_rad = np.radians(result.detector_skew)
                shear = np.tan(skew_rad)
                matrix = np.array([[1, 0], [-shear, 1]])
                offset = np.array([shear * rows / 2, 0])
                proj_b_corrected = affine_transform(proj_b_corrected, matrix, offset=offset, 
                                                     mode='constant', cval=0, order=3)
            if result.detector_tilt != 0.0:
                proj_b_corrected = cpu_rotate(proj_b_corrected, result.detector_tilt, 
                                               reshape=False, mode='constant', cval=0, order=3)
            shift_u = -result.cor_offset_u
            shift_v = -result.cor_offset_v
            if shift_u != 0.0 or shift_v != 0.0:
                proj_b_corrected = cpu_shift(proj_b_corrected, (shift_v, shift_u), 
                                              mode='constant', cval=0, order=3)
            
            self.visualizer.plot_projection_comparison(
                proj_a, proj_b_flip, proj_b_corrected,
                f'Projection Pair {pair[0]}/{pair[1]} Alignment'
            )
        
        # Phase 4: Generate outputs
        logger.info("\n" + "=" * 60)
        logger.info("Phase 4: Saving results...")
        logger.info("=" * 60)
        
        result.save(os.path.join(self.config.output_path, "calibration_result.json"))
        self.visualizer.generate_report(result, self.config, spatial_analysis)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("CALIBRATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"COR offset (u): {result.cor_offset_u:.4f} pixels")
        logger.info(f"COR offset (v): {result.cor_offset_v:.4f} pixels")
        logger.info(f"Detector tilt:  {result.detector_tilt:.4f} degrees")
        logger.info(f"Detector skew:  {result.detector_skew:.4f} degrees")
        logger.info(f"SOD correction: {result.sod_correction:.4f} mm")
        logger.info(f"Angular offset: {result.angular_offset:.4f} degrees")
        logger.info(f"Final cost:     {result.final_cost:.6f}")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {self.config.output_path}")
        
        return result


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CT Geometry Calibration Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory Structure:
    dataset_dir/
    ├── slices/
    │   ├── metadata.json    <- geometry and acquisition parameters
    │   ├── 0001.raw         <- projection files
    │   ├── 0002.raw
    │   └── ...
    └── geometry_calib/      <- calibration output (created automatically)
        ├── calibration_result.json
        ├── plots/
        └── report.html

Examples:
    python ct_geometry_calibration.py --dataset /path/to/my_scan
    python ct_geometry_calibration.py --dataset CBCT_3D_SheppLogan --method joint
    python ct_geometry_calibration.py --config custom_calibration.json
        """
    )
    
    parser.add_argument("--dataset", "-d", default="CBCT_3D_SheppLogan",
                        help="Path to dataset directory (default: CBCT_3D_SheppLogan)")
    parser.add_argument("--config", "-c", 
                        help="Path to full configuration JSON (overrides --dataset)")
    parser.add_argument("--method", "-m", default="sequential",
                        choices=["sequential", "joint", "coarse_to_fine"],
                        help="Optimization method (default: sequential)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    parser.add_argument("--downsample", "-ds", type=int, default=1,
                        help="Downsample factor for calibration (default: 1)")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = CalibrationConfig.from_json(args.config)
    else:
        logger.info(f"Loading dataset from {args.dataset}")
        config = CalibrationConfig.from_dataset(args.dataset)
    
    # Apply CLI overrides
    if args.no_gpu:
        config.use_gpu = False
    if args.downsample > 1:
        config.calibration_downsample = args.downsample
        logger.info(f"Calibration downsample factor: {args.downsample}")
    
    # Log GPU status
    if config.use_gpu and GPU_AVAILABLE:
        # logger.info(f"GPU acceleration: ENABLED ({cp.cuda.Device().name})")
        logger.info(f"GPU acceleration: ENABLED")
    else:
        logger.info("GPU acceleration: DISABLED")
    
    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)
    
    # Save configuration snapshot
    config.to_json(os.path.join(config.output_path, "calibration_config.json"))
    
    # Run pipeline
    pipeline = CalibrationPipeline(config)
    result = pipeline.run(method=args.method)
    
    print(f"\n{'='*60}")
    print("✓ Calibration complete!")
    print(f"{'='*60}")
    print(f"  Dataset:     {config.dataset_dir}")
    print(f"  Results:     {config.output_path}/")
    print(f"  COR (u):     {result.cor_offset_u:.3f} pixels")
    print(f"  COR (v):     {result.cor_offset_v:.3f} pixels")
    print(f"  Tilt:        {result.detector_tilt:.3f} degrees")
    print(f"  Final cost:  {result.final_cost:.6f}")
    print(f"{'='*60}")
    
    return result


if __name__ == "__main__":
    main()