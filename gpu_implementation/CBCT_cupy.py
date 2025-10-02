from networkx import volume
import numpy as np
import os
import time
import cupy as cp
import json
from PIL import Image
from tqdm import tqdm
import logging
import pickle
import glob
import tifffile
from dataclasses import dataclass, asdict, field
from typing import Tuple, Dict, Any, Optional, List
from open_pickled_result import view_pickled_volume_napari

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def discover_projection_files(folder: str, allowed_exts=(".tiff", ".tif", ".jpg", ".jpeg", ".png")) -> List[str]:
    pattern = os.path.join(folder, "*")
    files = [f for f in sorted(glob.glob(pattern)) if os.path.splitext(f)[1].lower() in allowed_exts]
    return files

"""
Example metadata.json file structure:
{
    "name": "Small_CBCT_Dataset",
    "num_projections": 36,
    "detector_pixels": [3052, 2500],
    "detector_size_mm": [430.0, 350.0],
    "detector_offset": [0.0, 0.0],
    "volume_voxels": [256, 256, 256],
    "volume_size_mm": [20.0, 20.0, 86.0],
    "source_origin_dist": 150.0,
    "source_detector_dist": 600.0,
    "angle_step": 10.0,
    "start_angle": 0.0,
    "projection_dtype": "uint16",
    "file_format": "jpg",
    "cosine_weighting": true,
    "filter_type": "none",
    "apply_log_correction": false,
    "apply_bad_pixel_correction": false,
    "apply_noise_reduction": false,
    "apply_truncation_correction": false,
    "truncation_width": 0,
    "dark_current": 0.0,
    "max_gpu_memory_fraction": 0.8,
    "save_intermediate": true
}
"""

@dataclass
class CBCTConfig:
    # Voxel/volume setup
    num_voxels: Tuple[int, int, int] = (256, 256, 256)
    volume_size_mm: Tuple[float, float, float] = (17.543, 17.543, 17.543)
    
    # Pixel/detector setup: detector_pixels = (nu, nv)
    detector_pixels: Tuple[int, int] = (2860, 2860)  # (nu, nv)
    detector_size_mm: Tuple[float, float] = (430.0, 350.0)
    detector_offset: Tuple[float, float] = (0.0, 0.0)
    
    # Acquisition geometry
    num_projections: int = 360
    angle_step: float = 1.0
    start_angle: float = 0.0            # <-- NEW, in degrees
    source_origin_dist: float = 212.515
    source_detector_dist: float = 1304.5
    
    # Projection data type (raw file reader)
    projection_dtype: str = 'uint16'    # <-- NEW
    
    # Preprocessing
    cosine_weighting: bool = True
    dark_current: float = 100.0
    bad_pixel_threshold: Optional[int] = 32768
    apply_log_correction: bool = True
    apply_bad_pixel_correction: bool = False
    apply_noise_reduction: bool = False
    apply_truncation_correction: bool = False
    truncation_width: int = 125
    
    # Filtering
    filter_type: str = 'none'
    
    # Saving
    save_intermediate: bool = True
    input_path: str = r"data\\20200225_AXI_final_code\\slices\\"
    intermediate_path: str = "data/20200225_AXI_final_code/intermediate"
    output_path: str = "data/20200225_AXI_final_code/results"
    
    # Other
    max_gpu_memory_fraction: float = 0.8

    # Internal state
    _derived_cache: Optional[Dict[str, Any]] = None
    _metadata_loaded: bool = False

    @classmethod
    def create_with_metadata(cls, folder: str, metadata_filename: str = "metadata.json") -> 'CBCTConfig':
        """
        Create CBCTConfig by loading metadata from folder.
        This is the primary way to create a config with metadata.
        """
        metadata_path = os.path.join(folder, metadata_filename)
        
        # Start with defaults
        config = cls()
        config.input_path = folder
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                config._apply_metadata(metadata)
                config._metadata_loaded = True
                
                logger.info(f"✓ Loaded metadata from {metadata_path}")
                logger.info(f"Dataset: {metadata.get('name', 'Unknown')}")
                logger.info(f"Config - Projections: {config.num_projections}, "
                          f"Detector: {config.detector_pixels}, "
                          f"Volume: {config.num_voxels}")
                
            except Exception as e:
                logger.error(f"✗ Failed to load metadata from {metadata_path}: {e}")
                raise RuntimeError(f"Critical: Could not load required metadata from {metadata_path}")
        else:
            logger.warning(f"⚠ No metadata file found at {metadata_path}")
            logger.warning("Using default configuration values")
            logger.info(f"Defaults - Projections: {config.num_projections}, "
                       f"Detector: {config.detector_pixels}, "
                       f"Volume: {config.num_voxels}")
        
        return config

    
    def _apply_metadata(self, metadata: Dict[str, Any]):
        # Acquisition parameters
        if 'angle_step' in metadata:
            self.angle_step = metadata['angle_step']
        if 'start_angle' in metadata:              # <-- NEW
            self.start_angle = metadata['start_angle']
        if 'num_projections' in metadata:
            self.num_projections = metadata['num_projections']
        
        # Detector parameters
        if 'detector_pixels' in metadata:
            self.detector_pixels = tuple(metadata['detector_pixels'])
        if 'detector_size_mm' in metadata:
            self.detector_size_mm = tuple(metadata['detector_size_mm'])
        if 'detector_offset' in metadata:
            self.detector_offset = tuple(metadata['detector_offset'])
        
        # Voxel/volume parameters
        if 'volume_voxels' in metadata:
            self.num_voxels = tuple(metadata['volume_voxels'])
        if 'volume_size_mm' in metadata:
            self.volume_size_mm = tuple(metadata['volume_size_mm'])

        # Distances
        if 'source_origin_dist' in metadata:
            self.source_origin_dist = metadata['source_origin_dist']
        if 'source_detector_dist' in metadata:
            self.source_detector_dist = metadata['source_detector_dist']
        
        # Projection dtype
        if 'projection_dtype' in metadata:         # <-- NEW
            self.projection_dtype = metadata['projection_dtype']
        
        # Processing
        if 'save_intermediate' in metadata:
            self.save_intermediate = metadata['save_intermediate']
        if 'max_gpu_memory_fraction' in metadata:
            self.max_gpu_memory_fraction = metadata['max_gpu_memory_fraction']

    @classmethod
    def from_metadata(cls, metadata: dict) -> 'CBCTConfig':
        """
        DEPRECATED: Use create_with_metadata() instead.
        Legacy method kept for backward compatibility.
        """
        logger.warning("from_metadata() is deprecated. Use create_with_metadata() instead.")
        config = cls()
        config._apply_metadata(metadata)
        config._metadata_loaded = True
        return config

    def derived(self) -> Dict[str, Any]:
        """
        Compute and cache derived parameters.
        Cache is invalidated when config changes.
        """
        if self._derived_cache is None:
            nx, ny, nz = self.num_voxels
            sx, sy, sz = self.volume_size_mm
            nu, nv = self.detector_pixels
            su, sv = self.detector_size_mm
            dx, dy, dz = sx / nx, sy / ny, sz / nz
            du, dv = su / nu, sv / nv

            # Convert offset from original detector pixels to binned mm coordinates
            offset_u_pixels_binned = self.detector_offset[0] / 4  # 1430.1/4 = 357.5
            offset_v_pixels_binned = self.detector_offset[1] / 4  # 1429.5/4 = 357.4

            # Convert to mm using BINNED pixel size
            offset_u_mm = (offset_u_pixels_binned - nu/2) * du  # Offset from center
            offset_v_mm = (offset_v_pixels_binned - nv/2) * dv
            
            self._derived_cache = {
                'param_dx': dx, 'param_dy': dy, 'param_dz': dz,
                'param_du': du, 'param_dv': dv,
                # 'param_us': np.arange((-nu/2 + 0.5), (nu/2), 1) * du + self.detector_offset[0],
                # 'param_vs': np.arange((-nv/2 + 0.5), (nv/2), 1) * dv + self.detector_offset[1],
                'param_us': (np.arange(nu) - nu/2 + 0.5) * du + offset_u_mm,
                'param_vs': (np.arange(nv) - nv/2 + 0.5) * dv + offset_v_mm,
                'param_xs': np.arange((-nx/2 + 0.5), (nx/2), 1) * dx,
                'param_ys': np.arange((-ny/2 + 0.5), (ny/2), 1) * dy,
                'param_zs': np.arange((-nz/2 + 0.5), (nz/2), 1) * dz,
            }
        
        return self._derived_cache

    def has_metadata(self) -> bool:
        """Check if metadata was successfully loaded"""
        return self._metadata_loaded

    def invalidate_cache(self) -> None:
        """Force recalculation of derived parameters on next access"""
        self._derived_cache = None


class CBCTDataLoader:
    def __init__(self, config: CBCTConfig):
        self.config = config
        
    @classmethod
    def from_folder(cls, folder: str, metadata_filename: str = "metadata.json") -> 'CBCTDataLoader':
        """Create CBCTDataLoader with metadata-aware config"""
        config = CBCTConfig.create_with_metadata(folder, metadata_filename)
        return cls(config)
        
    def _load_raw_file(self, filepath: str, shape: Tuple[int, int], dtype: str = 'uint16') -> np.ndarray:
        """Load RAW binary file with specified dimensions and data type"""
        dtype_map = {
            'uint8': (np.uint8, 1),
            'uint16': (np.uint16, 2), 
            'uint32': (np.uint32, 4),
            'float32': (np.float32, 4),
            'float64': (np.float64, 8)
        }
        
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
            
        np_dtype, bytes_per_pixel = dtype_map[dtype]
        expected_size = shape[0] * shape[1] * bytes_per_pixel
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"RAW file not found: {filepath}")
            
        file_size = os.path.getsize(filepath)
        if file_size != expected_size:
            logger.warning(f"RAW file size mismatch: {filepath} "
                         f"(expected {expected_size}, got {file_size} bytes)")
        
        try:
            with open(filepath, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np_dtype)
                return data.reshape(shape).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load RAW file {filepath}: {e}")
    
    def _load_tiff_file(self, filepath: str) -> np.ndarray:
        """Load TIFF file using tifffile for better format support"""
        try:
            # Try tifffile first (handles multi-page, various formats)
            data = tifffile.imread(filepath)
            if data.ndim == 3 and data.shape[0] == 1:
                data = data.squeeze(0)  # Remove singleton dimension
            # logger.info(f"Loaded TIFF {filepath} with shape {data.shape} and dtype {data.dtype}")
            return data.astype(np.float32)
        except Exception as e1:
            try:
                # Fallback to PIL
                with Image.open(filepath) as img:
                    return np.array(img.convert("F"))
            except Exception as e2:
                raise RuntimeError(f"Failed to load TIFF {filepath}. "
                                 f"tifffile error: {e1}, PIL error: {e2}")
    
    def _load_image_file(self, filepath: str) -> np.ndarray:
        """Load standard image formats (PNG, JPG, etc.)"""
        try:
            with Image.open(filepath) as img:
                return np.array(img.convert("F"))
        except Exception as e:
            raise RuntimeError(f"Failed to load image {filepath}: {e}")
    
    def _load_single_projection(self, filepath: str, 
                              raw_shape: Optional[Tuple[int, int]] = None,
                              raw_dtype: str = 'uint16') -> np.ndarray:
        """Load a single projection file based on extension"""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.raw':
            if raw_shape is None:
                raw_shape = self.config.detector_pixels[::-1]  # Use config detector size
            return self._load_raw_file(filepath, raw_shape, raw_dtype)
        elif ext in ['.tiff', '.tif']:
            return self._load_tiff_file(filepath)
        elif ext in ['.png', '.jpg', '.jpeg']:
            return self._load_image_file(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    # def load_projection_stack(self, folder: Optional[str] = None) -> 'cp.ndarray':
    #     """
    #     Load projection stack using the established config.
    #     No longer reloads metadata - config is immutable after creation.
    #     """
    #     if folder is None:
    #         folder = self.config.input_path
            
    #     # Validate that we're using the expected folder
    #     if folder != self.config.input_path and self.config.has_metadata():
    #         logger.warning(f"Loading from {folder} but metadata was loaded from {self.config.input_path}")
        
    #     # Rest of loading logic remains the same, but uses self.config consistently
    #     files = discover_projection_files(folder)
        
    #     if len(files) != self.config.num_projections:
    #         logger.warning(f"⚠ Found {len(files)} files, config expects {self.config.num_projections}")
            
    #     files_to_use = files[:self.config.num_projections]
    #     logger.info(f"Loading {len(files_to_use)} projection files using established config")


    def load_projection_stack(self, folder: Optional[str] = None, 
                            metadata_filename: str = "metadata.json") -> cp.ndarray:
        """
        Load projection stack from folder with JSON metadata
        
        Parameters:
        -----------
        folder : str, optional
            Directory containing projection files and JSON metadata. If None, uses config.input_path
        metadata_filename : str, default "metadata.json"
            Name of the JSON metadata file in the folder
        """
        # Use config path if folder not provided
        if folder is None:
            folder = self.config.input_path
            
        # Load and apply metadata from folder
        metadata_path = os.path.join(folder, metadata_filename)
        
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Update configuration with metadata values
                updated_config = CBCTConfig.from_metadata(metadata)
                updated_config.input_path = folder  # Preserve the folder path
                self.config = updated_config
                
                logger.info(f"Loaded and applied metadata from {metadata_path}")
                logger.info(f"Dataset: {metadata.get('name', 'Unknown')}")
                logger.info(f"Updated config - Projections: {self.config.num_projections}, "
                        f"Detector: {self.config.detector_pixels}, "
                        f"Volume: {self.config.num_voxels}")
                
            except Exception as e:
                logger.error(f"✗ Failed to load metadata from {metadata_path}: {e}")
                logger.error("Exiting due to metadata loading failure")
                raise RuntimeError(f"Critical: Could not load required metadata from {metadata_path}")
        else:
            logger.warning(f"⚠ No metadata file found at {metadata_path}")
            logger.warning("Continuing with default configuration values")
            logger.info(f"Using defaults - Projections: {self.config.num_projections}, "
                    f"Detector: {self.config.detector_pixels}, "
                    f"Volume: {self.config.num_voxels}")
        
        # Get files and validate
        files = discover_projection_files(folder)
        
        if len(files) == 0:
            logger.error(f"✗ No projection files found in {folder}")
            raise RuntimeError(f"No projection files found in {folder}")
            
        if len(files) != self.config.num_projections:
            logger.warning(f"⚠ Found {len(files)} files, config expects {self.config.num_projections}")
            if len(files) < self.config.num_projections:
                logger.warning(f"Using available {len(files)} files instead of {self.config.num_projections}")
            else:
                logger.warning(f"Using first {self.config.num_projections} of {len(files)} files")
                
        # Use available files, up to configured limit
        files_to_use = files[:self.config.num_projections]
        
        logger.info(f"Loading {len(files_to_use)} projection files from {folder}")
        logger.info(f"File format: {os.path.splitext(files_to_use[0])[1]}")
        logger.info(f"Expected detector shape: {self.config.detector_pixels[::-1]} (H x W)")
        
        # Load first file to determine dimensions and validate
        raw_shape = self.config.detector_pixels[::-1]  # (height, width)
        raw_dtype = getattr(self.config, 'projection_dtype', 'uint16')  # Get from updated config
        
        try:
            first_proj = self._load_single_projection(files_to_use[0], raw_shape, raw_dtype)
            expected_shape = self.config.detector_pixels[::-1]  # (height, width)
            
            if first_proj.shape != expected_shape:
                logger.warning(f"⚠ Projection shape {first_proj.shape} doesn't match "
                            f"config {expected_shape}. Will resize.")
        except Exception as e:
            logger.error(f"✗ Failed to load test projection {files_to_use[0]}: {e}")
            raise RuntimeError(f"Cannot load projection files from {folder}")
        
        # Pre-allocate and load all projections
        imgs = []
        
        for filepath in tqdm(files_to_use, desc="Loading projections"):
            try:
                proj = self._load_single_projection(filepath, raw_shape, raw_dtype)
                
                # Resize if needed
                if proj.shape != expected_shape:
                    from PIL import Image as PILImage
                    proj_pil = PILImage.fromarray(proj).resize(
                        (expected_shape[1], expected_shape[0]), 
                        PILImage.LANCZOS
                    )
                    proj = np.array(proj_pil, dtype=np.float32)
                
                imgs.append(proj)
                
            except Exception as e:
                logger.error(f"✗ Failed to load {filepath}: {e}")
                raise RuntimeError(f"Failed to load projection: {filepath}")
        
        # Stack and transfer to GPU
        stack = np.stack(imgs, axis=0)  # (nProj, height, width)
        logger.info(f"✓ Loaded projection stack shape: {stack.shape}")
        
        return cp.array(stack, dtype=cp.float32)
    
    def load_raw_dataset_info(self, info_file: str) -> Dict[str, Any]:
        """
        Load RAW dataset information from metadata file
        
        Expected format (JSON or simple key=value):
        {
            "width": 512,
            "height": 625,
            "dtype": "uint16",
            "num_projections": 360,
            "angle_increment": 1.0
        }
        """
        import json
        
        if not os.path.exists(info_file):
            logger.warning(f"Info file not found: {info_file}")
            return {}
            
        try:
            with open(info_file, 'r') as f:
                if info_file.endswith('.json'):
                    return json.load(f)
                else:
                    # Simple key=value format
                    info = {}
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            try:
                                info[key.strip()] = eval(value.strip())
                            except:
                                info[key.strip()] = value.strip()
                    return info
        except Exception as e:
            logger.warning(f"Failed to load dataset info from {info_file}: {e}")
            return {}

# Example usage:
"""
# Method 1: Load from metadata file directly
loader = CBCTDataLoader.from_metadata('dataset_metadata.json')
projections = loader.load_projection_stack()

# Method 2: Create config from metadata, then use loader
config = CBCTConfig.from_metadata('dataset_metadata.json')
loader = CBCTDataLoader(config)
projections = loader.load_projection_stack()

# Method 3: Traditional hardcoded approach (still supported)
# config = CBCTConfig(
#     num_voxels=(256, 256, 256),
#     volume_size_mm=(0.8, 50.0, 78.0),
#     detector_pixels=(625, 512),
#     detector_size_mm=(430.0, 350.0),
#     detector_offset=(0.0, 0.0),
#     num_projections=360,
#     angle_step=1.0,
#     source_origin_dist=212.515,
#     source_detector_dist=1304.5,
#     # ... other parameters
# )
# loader = CBCTDataLoader(config)
# projections = loader.load_projection_stack()
"""

class CBCTPreprocessor:
    def __init__(self, config: CBCTConfig):
        self.config = config
        self.derived = self.config.derived()  # Cache derived params at initialization

    def _update_derived_params(self):
        """Update derived parameters when config changes"""
        self.derived = self.config.derived()

    def _apply_dark_current_correction(self, proj: cp.ndarray) -> cp.ndarray:
        """Apply dark current correction: I_corrected = I_raw - dark_current"""
        if self.config.dark_current > 0:
            proj = proj - self.config.dark_current
            # Ensure no negative values after correction
            proj = cp.maximum(proj, 1.0)
        return proj
    
    def _apply_log_correction(self, proj: cp.ndarray, I0: float = None) -> cp.ndarray:
        """Apply logarithmic correction: proj = -log(I/I0)"""
        if I0 is None:
            # Use maximum intensity as reference if I0 not provided
            I0 = float(cp.max(proj))
        
        # Avoid division by zero and log(0)
        epsilon = 1e-6
        proj = cp.maximum(proj, epsilon)
        I0 = max(I0, epsilon)
        
        # Apply log correction: -log(I/I0) = log(I0) - log(I)
        proj = cp.log(I0) - cp.log(proj)
        
        # Handle any remaining invalid values
        proj = cp.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)
        
        return proj

    def _apply_bad_pixel_correction(self, proj: cp.ndarray) -> cp.ndarray:
        """GPU-accelerated bad pixel correction using median filter"""
        try:
            from cupyx.scipy.ndimage import median_filter
            bad_mask = proj > self.config.bad_pixel_threshold
            if cp.any(bad_mask):
                filtered = median_filter(proj, size=3)
                proj = cp.where(bad_mask, filtered, proj)
            return proj
        except ImportError:
            logger.warning("cupyx.scipy not available; skipping bad pixel correction")
            return proj
    
    def _apply_noise_reduction(self, proj: cp.ndarray) -> cp.ndarray:
        """GPU-accelerated Gaussian noise reduction"""
        try:
            from cupyx.scipy.ndimage import gaussian_filter
            proj = gaussian_filter(proj, sigma=1.5)
            return proj
        except ImportError:
            logger.warning("cupyx.scipy not available; skipping noise reduction")
            return proj
    
    def _apply_truncation_correction(self, proj: cp.ndarray) -> cp.ndarray:
        """GPU-accelerated truncation artifact correction"""
        width = self.config.truncation_width
        
        if width > 0 and width < proj.shape[1]:
            # Extend edges with mean values
            left_edge = cp.mean(proj[:, :10], axis=1, keepdims=True)
            proj[:, :width] = left_edge
            
            right_edge = cp.mean(proj[:, -10:], axis=1, keepdims=True)
            proj[:, -width:] = right_edge
        
        return proj

    def preprocess_stack(self, stack: cp.ndarray) -> cp.ndarray:
        """
        Apply full preprocessing pipeline to projection stack
        Order: Dark Current → Log Correction → Bad Pixel → Noise Reduction → 
               Truncation Correction → Cosine Weighting
        """
        nProj, nv, nu = stack.shape
        proj_pre = cp.empty_like(stack)
        
        # Prepare cosine weighting matrix if needed
        CW = 1.0
        if self.config.cosine_weighting:
            uu, vv = cp.meshgrid(cp.array(self.derived['param_us']), 
                               cp.array(self.derived['param_vs']))
            CW = (self.config.source_detector_dist / 
                  cp.sqrt((uu ** 2 + vv ** 2) + self.config.source_detector_dist ** 2))
        
        # Determine I0 for log correction if needed (use first projection max)
        I0 = None
        if self.config.apply_log_correction:
            # Apply dark current first to get clean reference
            first_proj = stack[0].copy()
            if self.config.dark_current > 0:
                first_proj = self._apply_dark_current_correction(first_proj)
            I0 = float(cp.max(first_proj))
            logger.info(f"Using I0 = {I0:.2f} for log correction")
        
        for i in tqdm(range(nProj), desc="Preprocessing projections"):
            proj = stack[i].copy()
            
            # Step 1: Dark current correction
            if self.config.dark_current > 0:
                proj = self._apply_dark_current_correction(proj)
            
            # Step 2: Logarithmic correction
            if self.config.apply_log_correction:
                proj = self._apply_log_correction(proj, I0)

            # Step 3: Bad pixel correction
            if self.config.apply_bad_pixel_correction:
                proj = self._apply_bad_pixel_correction(proj)
            
            # Step 4: Noise reduction
            if self.config.apply_noise_reduction:
                proj = self._apply_noise_reduction(proj)
            
            # Step 5: Truncation correction
            if self.config.apply_truncation_correction:
                proj = self._apply_truncation_correction(proj)
            
            # Step 6: Cosine weighting
            if self.config.cosine_weighting:
                proj = proj * CW
            
            proj_pre[i] = proj
        
        # Build status message
        steps_enabled = []
        if self.config.dark_current > 0:
            steps_enabled.append("Dark current")
        if self.config.apply_log_correction:
            steps_enabled.append("Log correction")
        if self.config.apply_bad_pixel_correction:
            steps_enabled.append("Bad pixel")
        if self.config.apply_noise_reduction:
            steps_enabled.append("Noise reduction")
        if self.config.apply_truncation_correction:
            steps_enabled.append("Truncation")
        if self.config.cosine_weighting:
            steps_enabled.append("Cosine weighting")
        
        logger.info(f"Preprocessing completed with: {', '.join(steps_enabled)}")
        
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
        self._update_derived_params()
        self._compile_backprojection_kernel()
        self._warmup_gpu()

    def _update_derived_params(self):
        """Update derived parameters when config changes"""
        self.derived = self.config.derived()

    def _compile_backprojection_kernel(self):
        """
        Corrected CUDA kernel matching ASTRA's cone beam geometry.
        
        Key differences from original:
        1. Proper detector coordinate mapping (no sign errors)
        2. Correct handling of detector offsets
        3. Proper bounds checking
        4. Correct weighting factor
        """
        kernel_source = r"""
        extern "C" __global__
        void backproj_kernel(const float* __restrict__ filtered,
                            float* __restrict__ volume,
                            const float* __restrict__ angles,
                            const int nProj, const int nDetRows, const int nDetCols,
                            const int nVoxX, const int nVoxY, const int nVoxZ,
                            const float voxSizeX, const float voxSizeY, const float voxSizeZ,
                            const float detPixelU, const float detPixelV,
                            const float detOffsetU, const float detOffsetV,
                            const float sourceDist, const float detectorDist)
        {
            // Thread indices map to voxel coordinates
            int ix = blockDim.x * blockIdx.x + threadIdx.x;
            int iy = blockDim.y * blockIdx.y + threadIdx.y;
            int iz = blockDim.z * blockIdx.z + threadIdx.z;
            
            if (ix >= nVoxX || iy >= nVoxY || iz >= nVoxZ) return;
            
            // Voxel position in world coordinates (centered at origin)
            float vx = (ix - nVoxX / 2.0f + 0.5f) * voxSizeX;
            float vy = (iy - nVoxY / 2.0f + 0.5f) * voxSizeY;
            float vz = (iz - nVoxZ / 2.0f + 0.5f) * voxSizeZ;
            
            float accumulator = 0.0f;
            
            // Loop over all projections
            for (int proj = 0; proj < nProj; proj++) {
                float angle = angles[proj];
                float cosA = cosf(angle);
                float sinA = sinf(angle);
                
                // Rotate voxel position to projection coordinate system
                // Source is at (-sourceDist, 0) in rotated coordinates
                float px = vx * cosA + vy * sinA;
                float py = -vx * sinA + vy * cosA;
                float pz = vz;
                
                // Distance from source to voxel
                float sourceToVoxelY = py + sourceDist;
                
                // Skip if voxel is behind source
                if (sourceToVoxelY <= 0.0f) continue;
                
                // Project onto detector plane
                // Detector is at (detectorDist - sourceDist, 0) from source
                float projectionScale = detectorDist / sourceToVoxelY;
                
                // Detector coordinates (physical units, centered at principal point)
                float detU = px * projectionScale - detOffsetU;
                float detV = pz * projectionScale - detOffsetV;
                
                // Convert to pixel indices (detector center is at detector dimensions/2)
                float pixU = detU / detPixelU + nDetCols / 2.0f;
                float pixV = detV / detPixelV + nDetRows / 2.0f;
                
                // Bounds check with margin for interpolation
                if (pixU < 0.0f || pixU >= nDetCols - 1.0f || 
                    pixV < 0.0f || pixV >= nDetRows - 1.0f) continue;
                
                // Bilinear interpolation
                int u0 = (int)floorf(pixU);
                int v0 = (int)floorf(pixV);
                int u1 = u0 + 1;
                int v1 = v0 + 1;
                
                float fu = pixU - u0;
                float fv = pixV - v0;
                
                // Get projection data
                int baseIdx = proj * nDetRows * nDetCols;
                float p00 = filtered[baseIdx + v0 * nDetCols + u0];
                float p01 = filtered[baseIdx + v0 * nDetCols + u1];
                float p10 = filtered[baseIdx + v1 * nDetCols + u0];
                float p11 = filtered[baseIdx + v1 * nDetCols + u1];
                
                // Interpolate
                float val = (1.0f - fu) * (1.0f - fv) * p00 +
                           fu * (1.0f - fv) * p01 +
                           (1.0f - fu) * fv * p10 +
                           fu * fv * p11;
                
                // Distance weighting (cone beam geometry)
                float weight = (sourceDist * sourceDist) / 
                              (sourceToVoxelY * sourceToVoxelY);
                
                accumulator += val * weight;
            }
            
            // Average over projections and apply angle scaling
            float angleWeight = 2.0f * 3.14159265359f / nProj;
            volume[iz * nVoxX * nVoxY + iy * nVoxX + ix] = accumulator * angleWeight;
        }
        """
        self.backproj_kernel = cp.RawKernel(kernel_source, "backproj_kernel")
    
    def _warmup_gpu(self):
        """Perform GPU warmup to initialize CUDA context and compile kernel"""
        logger.info("Performing GPU warmup...")
        try:
            # Create small dummy data
            dummy_nProj, dummy_nv, dummy_nu = 4, 32, 32
            dummy_nx, dummy_ny, dummy_nz = 16, 16, 16
            
            dummy_filtered = cp.ones((dummy_nProj, dummy_nv, dummy_nu), dtype=cp.float32)
            dummy_volume = cp.zeros((dummy_nz, dummy_ny, dummy_nx), dtype=cp.float32)
            dummy_angles = cp.linspace(0, cp.pi, dummy_nProj, dtype=cp.float32)
            
            block_size = (4, 4, 4)
            grid_size = ((dummy_nx + 3) // 4, (dummy_ny + 3) // 4, (dummy_nz + 3) // 4)
            
            self.backproj_kernel(
                grid_size, block_size,
                (dummy_filtered.ravel(),
                 dummy_volume.ravel(),
                 dummy_angles,
                 cp.int32(dummy_nProj), cp.int32(dummy_nv), cp.int32(dummy_nu),
                 cp.int32(dummy_nx), cp.int32(dummy_ny), cp.int32(dummy_nz),
                 cp.float32(1.0), cp.float32(1.0), cp.float32(1.0),
                 cp.float32(1.0), cp.float32(1.0),
                 cp.float32(0.0), cp.float32(0.0),
                 cp.float32(50.0), cp.float32(100.0))
            )
            
            cp.cuda.Device().synchronize()
            logger.info("GPU warmup completed successfully")
            
        except Exception as e:
            logger.warning(f"GPU warmup failed: {e}")

    def backproject(self, filtered: cp.ndarray) -> np.ndarray:
        """
        Perform cone beam backprojection matching ASTRA's geometry.
        
        Args:
            filtered: Preprocessed and filtered projections (nProj, nDetRows, nDetCols)
        
        Returns:
            Reconstructed volume as numpy array (nz, ny, nx)
        """
        logger.info("Starting CUDA-accelerated backprojection")
        
        nProj, nDetRows, nDetCols = filtered.shape
        nVoxX, nVoxY, nVoxZ = self.config.num_voxels
        
        # Volume voxel sizes
        volSizeX, volSizeY, volSizeZ = self.config.volume_size_mm
        voxSizeX = volSizeX / nVoxX
        voxSizeY = volSizeY / nVoxY
        voxSizeZ = volSizeZ / nVoxZ
        
        # Detector pixel sizes
        detSizeU, detSizeV = self.config.detector_size_mm
        detPixelU = detSizeU / nDetCols
        detPixelV = detSizeV / nDetRows
        
        # Detector offsets (convert from pixel units if necessary)
        # detOffsetU, detOffsetV = self.config.detector_offset
        detOffsetU = 357.5 * detPixelU  # Convert pixels to mm
        detOffsetV = 357.4 * detPixelV  # Convert pixels to mm
        
        # Geometry distances
        sourceDist = self.config.source_origin_dist
        detectorDist = self.config.source_detector_dist
        
        # Compute projection angles
        start_angle_rad = np.deg2rad(self.config.start_angle)
        angle_step_rad = np.deg2rad(self.config.angle_step)
        angles = cp.array(
            [start_angle_rad + i * angle_step_rad for i in range(nProj)],
            dtype=cp.float32
        )
        
        # Initialize volume
        volume = cp.zeros((nVoxZ, nVoxY, nVoxX), dtype=cp.float32)
        
        # Ensure filtered projections are contiguous
        filtered_contiguous = cp.ascontiguousarray(filtered)
        
        # Launch kernel
        block_size = (8, 8, 8)
        grid_size = (
            (nVoxX + block_size[0] - 1) // block_size[0],
            (nVoxY + block_size[1] - 1) // block_size[1],
            (nVoxZ + block_size[2] - 1) // block_size[2]
        )
        
        logger.info(f"Kernel config - Grid: {grid_size}, Block: {block_size}")
        logger.info(f"Geometry - Source dist: {sourceDist:.2f}mm, "
                   f"Detector dist: {detectorDist:.2f}mm")
        logger.info(f"Voxel size: ({voxSizeX:.4f}, {voxSizeY:.4f}, {voxSizeZ:.4f}) mm")
        logger.info(f"Detector pixel: ({detPixelU:.4f}, {detPixelV:.4f}) mm")
        logger.info(f"Detector offset: ({detOffsetU:.2f}, {detOffsetV:.2f}) mm")
        
        try:
            self.backproj_kernel(
                grid_size, block_size,
                (filtered_contiguous.ravel(),
                 volume.ravel(),
                 angles,
                 cp.int32(nProj), cp.int32(nDetRows), cp.int32(nDetCols),
                 cp.int32(nVoxX), cp.int32(nVoxY), cp.int32(nVoxZ),
                 cp.float32(voxSizeX), cp.float32(voxSizeY), cp.float32(voxSizeZ),
                 cp.float32(detPixelU), cp.float32(detPixelV),
                 cp.float32(detOffsetU), cp.float32(detOffsetV),
                 cp.float32(sourceDist), cp.float32(detectorDist))
            )
            cp.cuda.Device().synchronize()
            
            # Debug output
            vol_np = cp.asnumpy(volume)
            logger.info(f"Volume stats - min: {vol_np.min():.6f}, "
                       f"max: {vol_np.max():.6f}, "
                       f"mean: {vol_np.mean():.6f}, "
                       f"std: {vol_np.std():.6f}")
            
            non_zero = np.count_nonzero(vol_np)
            total = vol_np.size
            logger.info(f"Non-zero voxels: {non_zero}/{total} ({100*non_zero/total:.2f}%)")
            
        except Exception as e:
            logger.error(f"CUDA backprojection kernel failed: {e}")
            raise
            
        logger.info(f"CUDA backprojection completed, volume shape: {volume.shape}")
        return cp.asnumpy(volume)

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

def CBCTPipeline(main_path: str):
    config = CBCTConfig.create_with_metadata(os.path.join(main_path, 'slices'), 'metadata.json')
    os.makedirs(config.output_path, exist_ok=True)
    os.makedirs(config.intermediate_path, exist_ok=True)
    data_loader = CBCTDataLoader(config)
    preprocessor = CBCTPreprocessor(config)
    ramp_filter = CBCTRampFilter(config)
    backprojector = CBCTBackprojector(config)
    t0 = time.time()
    projections = data_loader.load_projection_stack(config.input_path)
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
    main_path = 'data/20240530_ITRI_downsampled_4x'
    CBCTPipeline(main_path)
    view_pickled_volume_napari(path=os.path.join(main_path, 'results/volume.pickle'))
