#!/usr/bin/env python3
"""
CBCT Reconstruction using ASTRA Toolbox
Architecture: DataLoader -> Preprocessor -> ASTRAReconstructor
Adapted for the provided CT dataset metadata with proper cone beam geometry
"""

import numpy as np
import os
import time
import logging
from PIL import Image
from tqdm import tqdm
import traceback
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import astra

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ASTRACBCTConfig:
    # Dataset parameters from metadata
    num_projections: int = 1600
    scan_angle: float = 360.0
    start_angle: float = 270.0
    detector_size: Tuple[int, int] = (2860, 2860)  # (rows, cols)
    pixel_size_u: float = 0.15  # mm - detector u spacing
    pixel_size_v: float = 0.15  # mm - detector v spacing
    voxel_size: float = 0.006134010138512811  # mm
    source_object_dist: float = 28.625365287711134  # mm
    source_detector_dist: float = 699.9996522369905  # mm
    detector_offset_u: float = 1430.1098329145173  # pixels
    detector_offset_v: float = 1429.4998776624227  # pixels
    
    # Processing parameters
    dark_current: float = 100.0
    bad_pixel_threshold: int = 32768
    apply_log_correction: bool = True
    apply_bad_pixel_correction: bool = True
    apply_noise_reduction: bool = True
    apply_truncation_correction: bool = True
    truncation_width: int = 500
    
    # Reconstruction parameters
    volume_size: Tuple[int, int, int] = (512, 512, 512)  # Reduced for memory
    algorithm: str = "FDK_CUDA"  # or "CGLS3D_CUDA", "SIRT3D_CUDA"
    iterations: int = 50  # for iterative algorithms
    
    # Memory management
    projection_batch_size: int = 50
    downsample_factor: int = 4  # Reduce detector resolution for memory


class CBCTDataLoader:
    """Load and preprocess CBCT projection data"""
    
    def __init__(self, config: ASTRACBCTConfig):
        self.config = config
    
    def load_projections(self, proj_folder: str, start_num: int = 0) -> np.ndarray:
        """Load projection images and return as numpy array (angles, rows, cols)"""
        logger.info(f"Loading {self.config.num_projections} projections from {proj_folder}")
        
        # Check first projection for dimensions
        first_file = os.path.join(proj_folder, f"Projection_{start_num:04d}.tiff")
        if not os.path.exists(first_file):
            raise FileNotFoundError(f"First projection file not found: {first_file}")
        
        first_proj = np.array(Image.open(first_file))
        original_h, original_w = first_proj.shape
        logger.info(f"Original projection size: {original_h} x {original_w}")
        
        # Apply downsampling if needed
        if self.config.downsample_factor > 1:
            ds = self.config.downsample_factor
            new_h = original_h // ds
            new_w = original_w // ds
            logger.info(f"Downsampling to: {new_h} x {new_w}")
        else:
            new_h, new_w = original_h, original_w
        
        # Initialize projection array
        projections = np.zeros((self.config.num_projections, new_h, new_w), dtype=np.float32)
        
        # Load projections in batches
        batch_size = self.config.projection_batch_size
        num_batches = (self.config.num_projections + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Loading projection batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, self.config.num_projections)
            
            for i in range(batch_start, batch_end):
                proj_file = os.path.join(proj_folder, f"Projection_{start_num + i:04d}.tiff")
                
                if os.path.exists(proj_file):
                    try:
                        proj = np.array(Image.open(proj_file), dtype=np.float32)
                        
                        # Downsample if needed
                        if self.config.downsample_factor > 1:
                            ds = self.config.downsample_factor
                            proj = proj[::ds, ::ds]
                        
                        projections[i] = proj
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {proj_file}: {e}")
                        projections[i] = np.zeros((new_h, new_w), dtype=np.float32)
                else:
                    logger.warning(f"Missing projection: {proj_file}")
                    projections[i] = np.zeros((new_h, new_w), dtype=np.float32)
        
        logger.info(f"Loaded projections with shape: {projections.shape}")
        return projections


class CBCTPreprocessor:
    """Preprocess CBCT projections (dark current, log correction, etc.)"""
    
    def __init__(self, config: ASTRACBCTConfig):
        self.config = config
    
    def preprocess_projections(self, projections: np.ndarray) -> np.ndarray:
        """Apply all preprocessing steps to projection data"""
        logger.info("Preprocessing projections...")
        
        processed = projections.copy()
        
        # Dark current subtraction
        processed = processed - self.config.dark_current
        processed = np.maximum(processed, 1.0)  # Avoid log(0)
        
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
        """Apply logarithmic correction for Beer-Lambert law"""
        I0 = np.max(projections, axis=(1, 2), keepdims=True)  # Max per projection
        return -np.log(projections / I0)
    
    def _bad_pixel_correction(self, projections: np.ndarray) -> np.ndarray:
        """Simple bad pixel correction using median filter"""
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
        """Truncation artifact correction"""
        corrected = projections.copy()
        width = self.config.truncation_width
        
        if width > 0 and width < projections.shape[2]:
            # Extend edges with mean values
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
        """Check ASTRA installation and CUDA availability"""
        logger.info(f"ASTRA version: {astra.__version__}")
        
        if astra.astra.use_cuda():
            logger.info("CUDA acceleration available")
        else:
            logger.warning("CUDA not available, using CPU")
            # Fallback to CPU algorithms
            if self.config.algorithm.endswith("_CUDA"):
                self.config.algorithm = self.config.algorithm.replace("_CUDA", "")
    
    def create_geometry(self, projections: np.ndarray) -> Tuple[Dict, Dict]:
        """Create ASTRA projection and volume geometries"""
        num_angles, det_rows, det_cols = projections.shape
        
        # Projection geometry (cone beam)
        angles = np.linspace(
            np.radians(self.config.start_angle),
            np.radians(self.config.start_angle + self.config.scan_angle),
            num_angles,
            endpoint=False
        )
        
        # Adjust parameters for downsampling
        ds = self.config.downsample_factor
        pixel_size_u = self.config.pixel_size_u * ds
        pixel_size_v = self.config.pixel_size_v * ds
        detector_offset_u = self.config.detector_offset_u / ds
        detector_offset_v = self.config.detector_offset_v / ds
        
        # Create cone beam geometry
        proj_geom = astra.create_proj_geom(
            'cone',
            pixel_size_v,  # detector spacing in v direction
            pixel_size_u,  # detector spacing in u direction
            det_rows,      # number of detector rows
            det_cols,      # number of detector columns
            angles,        # projection angles
            self.config.source_object_dist,     # source to origin distance
            self.config.source_detector_dist - self.config.source_object_dist,  # origin to detector
            detector_offset_u - det_cols/2,     # detector offset u
            detector_offset_v - det_rows/2      # detector offset v
        )
        
        # Volume geometry
        vol_x, vol_y, vol_z = self.config.volume_size
        vol_geom = astra.create_vol_geom(
            vol_y, vol_x, vol_z,  # ASTRA uses y,x,z order
            -vol_x/2 * self.config.voxel_size, vol_x/2 * self.config.voxel_size,
            -vol_y/2 * self.config.voxel_size, vol_y/2 * self.config.voxel_size,
            -vol_z/2 * self.config.voxel_size, vol_z/2 * self.config.voxel_size
        )
        
        logger.info(f"Created geometries - Projections: {proj_geom}, Volume: {vol_geom}")
        return proj_geom, vol_geom
    
    def reconstruct(self, projections: np.ndarray) -> np.ndarray:
        """Perform CBCT reconstruction using ASTRA"""
        logger.info(f"Starting ASTRA reconstruction with algorithm: {self.config.algorithm}")
        
        try:
            # Create geometries
            proj_geom, vol_geom = self.create_geometry(projections)

            # Rearrange projections to match ASTRA's expected shape
            projections = projections.transpose(1, 0, 2)  # to (rows, angles, cols)

            # Create ASTRA data objects
            proj_id = astra.data3d.create('-proj3d', proj_geom, projections)
            vol_id = astra.data3d.create('-vol', vol_geom)
            
            # Configure reconstruction algorithm
            cfg = astra.astra_dict(self.config.algorithm)
            cfg['ReconstructionDataId'] = vol_id
            cfg['ProjectionDataId'] = proj_id
            
            # Algorithm-specific settings
            if 'FDK' in self.config.algorithm:
                # FDK is direct reconstruction
                pass
            else:
                # Iterative algorithms
                cfg['option'] = {
                    'MinConstraint': 0,  # Non-negativity constraint
                }
                if hasattr(self.config, 'iterations'):
                    cfg['option']['MaxIter'] = self.config.iterations
            
            # Create and run algorithm
            alg_id = astra.algorithm.create(cfg)
            
            start_time = time.perf_counter()
            
            if 'FDK' in self.config.algorithm:
                # Direct reconstruction
                astra.algorithm.run(alg_id)
            else:
                # Iterative reconstruction with progress
                for i in tqdm(range(self.config.iterations), desc="Reconstructing"):
                    astra.algorithm.run(alg_id, 1)
            
            end_time = time.perf_counter()
            reconstruction_time = end_time - start_time
            
            # Get reconstruction result
            reconstruction = astra.data3d.get(vol_id)
            
            # Clean up ASTRA objects
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(proj_id)
            astra.data3d.delete(vol_id)
            
            logger.info(f"Reconstruction completed in {reconstruction_time:.2f} seconds")
            logger.info(f"Reconstruction shape: {reconstruction.shape}")
            logger.info(f"Value range: [{np.min(reconstruction):.4f}, {np.max(reconstruction):.4f}]")
            
            return reconstruction
            
        except Exception as e:
            logger.error(f"ASTRA reconstruction failed: {e}")
            traceback.print_exc()
            raise
    
    def save_results(self, reconstruction: np.ndarray, output_folder: str, 
                    reconstruction_time: float) -> Dict[str, Any]:
        """Save reconstruction results"""
        logger.info("Saving reconstruction results...")
        os.makedirs(output_folder, exist_ok=True)
        
        # Normalize and convert to 16-bit
        recon_norm = (reconstruction - np.min(reconstruction)) / np.ptp(reconstruction)
        recon_16bit = (recon_norm * 65535).astype(np.uint16)
        
        # Save raw volume
        output_raw_path = os.path.join(output_folder, 'astra_volume.raw')
        recon_16bit.tofile(output_raw_path)
        logger.info(f"Saved raw volume: {output_raw_path}")
        
        # Save metadata
        metadata = {
            'shape': reconstruction.shape,
            'voxel_size': self.config.voxel_size,
            'data_type': 'uint16',
            'reconstruction_time': reconstruction_time,
            'algorithm': self.config.algorithm,
            'downsample_factor': self.config.downsample_factor,
            'config': self.config.__dict__
        }
        metadata_path = os.path.join(output_folder, 'astra_metadata.npz')
        np.savez(metadata_path, **metadata)
        logger.info(f"Saved metadata: {metadata_path}")
        
        # Save middle slices as preview images
        mid_z = reconstruction.shape[0] // 2
        mid_y = reconstruction.shape[1] // 2
        mid_x = reconstruction.shape[2] // 2
        
        # XY slice (axial)
        slice_xy = recon_norm[mid_z, :, :]
        slice_xy_img = Image.fromarray((slice_xy * 255).astype(np.uint8))
        xy_path = os.path.join(output_folder, 'slice_axial.png')
        slice_xy_img.save(xy_path)
        
        # XZ slice (coronal)
        slice_xz = recon_norm[:, mid_y, :]
        slice_xz_img = Image.fromarray((slice_xz * 255).astype(np.uint8))
        xz_path = os.path.join(output_folder, 'slice_coronal.png')
        slice_xz_img.save(xz_path)
        
        # YZ slice (sagittal)
        slice_yz = recon_norm[:, :, mid_x]
        slice_yz_img = Image.fromarray((slice_yz * 255).astype(np.uint8))
        yz_path = os.path.join(output_folder, 'slice_sagittal.png')
        slice_yz_img.save(yz_path)
        
        logger.info(f"Saved preview slices: {xy_path}, {xz_path}, {yz_path}")
        
        return {
            'reconstruction': reconstruction,
            'shape': reconstruction.shape,
            'reconstruction_time': reconstruction_time,
            'output_folder': output_folder,
            'raw_file': output_raw_path,
            'metadata_file': metadata_path,
            'preview_files': [xy_path, xz_path, yz_path]
        }


def main():
    """Main reconstruction pipeline"""
    
    # # Configuration based on provided metadata
    # config = ASTRACBCTConfig(
    #     num_projections=1600,
    #     scan_angle=360.0,
    #     start_angle=270.0,
    #     detector_size=(2860, 2860),
    #     pixel_size_u=0.15,
    #     pixel_size_v=0.15,
    #     voxel_size=0.006134010138512811,
    #     # voxel_size=0.01713,  # Increased voxel size for larger FOV
    #     source_object_dist=28.625365287711134,
    #     source_detector_dist=699.9996522369905,
    #     detector_offset_u=1430.1098329145173 / 0.01713,
    #     detector_offset_v=1429.4998776624227 / 0.01713,

    #     # Reduced settings for memory management
    #     volume_size=(1024, 1024, 1024),  # Smaller volume
    #     algorithm="FDK_CUDA",
    #     projection_batch_size=50,
    #     downsample_factor=2,  # Reduce from 2860x2860 to 1430x1430
        
    #     # Processing options
    #     apply_log_correction=True,
    #     apply_bad_pixel_correction=True,
    #     apply_noise_reduction=True,
    #     apply_truncation_correction=True,
    #     truncation_width=125  # Adjusted for downsampling
    # )
    
    # Configuration based on provided metadata
    config = ASTRACBCTConfig(
        num_projections=1600,
        scan_angle=360.0,
        start_angle=270.0,
        detector_size=(715, 715),
        pixel_size_u=0.6,
        pixel_size_v=0.6,
        voxel_size=0.006134010138512811,
        # voxel_size=0.01713,  # Increased voxel size for larger FOV
        source_object_dist=28.625365287711134,
        source_detector_dist=699.9996522369905,
        detector_offset_u=357.5 / 0.01713,
        detector_offset_v=357.4 / 0.01713,

        # Reduced settings for memory management
        volume_size=(512, 512, 512),  # Smaller volume
        algorithm="FDK_CUDA",
        projection_batch_size=50,
        downsample_factor=1, 
        
        # Processing options
        apply_log_correction=True,
        apply_bad_pixel_correction=True,
        apply_noise_reduction=True,
        apply_truncation_correction=True,
        truncation_width=125  # Adjusted for downsampling
    )

    # Input/output paths
    projection_folder = 'data/20240530_ITRI_downsampled_4x/slices/'
    output_folder = 'data/astra_reconstructed/'
    
    if not os.path.exists(projection_folder):
        logger.error(f"Projection folder not found: {projection_folder}")
        logger.info("Please create the folder and place TIFF projection files in it")
        return 1
    
    try:
        # Initialize components
        data_loader = CBCTDataLoader(config)
        preprocessor = CBCTPreprocessor(config)
        reconstructor = ASTRAReconstructor(config)
        
        # Pipeline execution
        logger.info("=== ASTRA CBCT Reconstruction Pipeline ===")
        
        # 1. Load projections
        projections = data_loader.load_projections(projection_folder)
        
        # 2. Preprocess projections
        # projections_processed = preprocessor.preprocess_projections(projections)
        projections_processed = projections  # Skip preprocessing for testing

        # 3. Reconstruct volume
        start_total = time.perf_counter()
        reconstruction = reconstructor.reconstruct(projections_processed)
        end_total = time.perf_counter()
        total_time = end_total - start_total
        
        # 4. Save results
        results = reconstructor.save_results(reconstruction, output_folder, total_time)
        
        # Summary
        logger.info("=== Reconstruction Complete ===")
        logger.info(f"Algorithm: {config.algorithm}")
        logger.info(f"Volume shape: {results['shape']}")
        logger.info(f"Total time: {results['reconstruction_time']:.2f} seconds")
        logger.info(f"Results saved in: {results['output_folder']}")
        logger.info(f"Voxel size: {config.voxel_size:.6f} mm")
        logger.info(f"Downsample factor: {config.downsample_factor}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Reconstruction interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())