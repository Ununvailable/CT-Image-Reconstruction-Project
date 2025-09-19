import numpy as np
import os
import glob
from pathlib import Path
from PIL import Image
import time
import cupy as cp
from tqdm import tqdm
import traceback
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CBCTConfig:
    """CBCT reconstruction configuration"""
    num_projections: int = 1600
    scan_angle: float = 360.0
    start_angle: float = 270.0
    detector_size: Tuple[int, int] = (2860, 2860)
    pixel_size: float = 0.15  # mm
    voxel_size: float = 0.006134010138512811  # mm
    source_object_dist: float = 28.625365287711134  # mm
    source_detector_dist: float = 699.9996522369905  # mm
    detector_offset_u: float = 1430.1098329145173
    detector_offset_v: float = 1429.4998776624227
    
    # Processing parameters
    dark_current: float = 100.0
    bad_pixel_threshold: int = 32768
    apply_log_correction: bool = True
    apply_bad_pixel_correction: bool = True
    apply_noise_reduction: bool = True
    apply_truncation_correction: bool = True
    truncation_width: int = 500
    
    # Volume denoising
    volume_denoising_enabled: bool = True
    denoising_iterations: int = 4
    denoising_sigma: float = 0.7
    
    # Memory management
    projection_batch_size: int = 50
    filter_chunk_size: int = 100
    max_gpu_memory_fraction: float = 0.8

class GPUMemoryManager:
    """GPU memory management utilities"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current GPU memory information in GB"""
        mempool = cp.get_default_memory_pool()
        device_info = cp.cuda.Device().mem_info
        
        return {
            'used': mempool.used_bytes() / 1e9,
            'pool_total': mempool.total_bytes() / 1e9,
            'device_free': device_info[0] / 1e9,
            'device_total': device_info[1] / 1e9
        }
    
    @staticmethod
    def cleanup():
        """Clean up GPU memory pools"""
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    
    @staticmethod
    def estimate_sinogram_memory(shape: Tuple[int, int]) -> Dict[str, float]:
        """Estimate memory requirements for sinogram processing"""
        projLen, numAngles = shape
        
        sino_mem = projLen * numAngles * 4 / 1e9  # float32
        fft_temp = projLen * numAngles * 8 / 1e9  # complex64
        backproj_mem = projLen * projLen * 4 / 1e9
        total = sino_mem + fft_temp + backproj_mem
        
        return {
            'sinogram': sino_mem,
            'fft_temporary': fft_temp,
            'backprojection': backproj_mem,
            'total': total
        }

class CBCTPreprocessor:
    """CBCT projection preprocessing"""
    
    def __init__(self, config: CBCTConfig):
        self.config = config
    
    def preprocess_projection_gpu(self, proj_gpu: cp.ndarray) -> cp.ndarray:
        """GPU preprocessing for single CBCT projection"""
        try:
            # Dark current correction
            proj_processed = proj_gpu.astype(cp.float32) - self.config.dark_current
            proj_processed = cp.maximum(proj_processed, 1.0)
            
            # Logarithmic correction
            if self.config.apply_log_correction:
                I0 = cp.max(proj_processed)
                proj_processed = -cp.log(proj_processed / I0)
            
            # Bad pixel correction
            if self.config.apply_bad_pixel_correction:
                proj_processed = self._bad_pixel_correction_gpu(proj_processed)
            
            # Noise reduction
            if self.config.apply_noise_reduction:
                proj_processed = self._noise_reduction_gpu(proj_processed)
            
            # Truncation correction
            if self.config.apply_truncation_correction:
                proj_processed = self._truncation_correction_gpu(proj_processed)
            
            return proj_processed
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def _bad_pixel_correction_gpu(self, proj_gpu: cp.ndarray) -> cp.ndarray:
        """GPU bad pixel correction using median filter"""
        try:
            from cupyx.scipy.ndimage import median_filter
            
            bad_mask = proj_gpu > self.config.bad_pixel_threshold
            
            if cp.any(bad_mask):
                filtered = median_filter(proj_gpu, size=3)
                return cp.where(bad_mask, filtered, proj_gpu)
            
            return proj_gpu
        except ImportError:
            logger.warning("cupyx not available, skipping bad pixel correction")
            return proj_gpu
    
    def _noise_reduction_gpu(self, proj_gpu: cp.ndarray) -> cp.ndarray:
        """Simple GPU noise reduction"""
        try:
            from cupyx.scipy.ndimage import gaussian_filter
            return gaussian_filter(proj_gpu, sigma=1.5)
        except ImportError:
            logger.warning("cupyx not available, skipping noise reduction")
            return proj_gpu
    
    def _truncation_correction_gpu(self, proj_gpu: cp.ndarray) -> cp.ndarray:
        """Apply truncation correction by extending edges"""
        H, W = proj_gpu.shape
        corrected = proj_gpu.copy()
        
        width = self.config.truncation_width
        
        # Left truncation correction
        if width > 0 and width < W:
            left_edge = corrected[:, :10].mean(axis=1, keepdims=True)
            corrected[:, :width] = left_edge
        
        # Right truncation correction
        if width > 0 and width < W:
            right_edge = corrected[:, -10:].mean(axis=1, keepdims=True)
            corrected[:, -width:] = right_edge
        
        return corrected

class CBCTDataLoader:
    """CBCT projection data loading"""
    
    def __init__(self, config: CBCTConfig):
        self.config = config
        self.preprocessor = CBCTPreprocessor(config)
    
    def load_projections(self, proj_folder: str, start_num: int = 0) -> cp.ndarray:
        """Load and preprocess CBCT projections with GPU acceleration"""
        logger.info(f"Loading {self.config.num_projections} CBCT projections from {proj_folder}")
        
        # Validate first projection
        first_file = os.path.join(proj_folder, f"Projection_{start_num:04d}.tiff")
        if not os.path.exists(first_file):
            raise FileNotFoundError(f"First projection file not found: {first_file}")
        
        first_proj = np.array(Image.open(first_file))
        H, W = first_proj.shape
        logger.info(f"Projection size: {H} x {W}")
        
        # Check memory requirements
        sinogram_shape = (H, self.config.num_projections)
        memory_est = GPUMemoryManager.estimate_sinogram_memory(sinogram_shape)
        memory_info = GPUMemoryManager.get_memory_info()
        
        logger.info(f"Estimated sinogram memory: {memory_est['sinogram']:.2f} GB")
        logger.info(f"Available GPU memory: {memory_info['device_free']:.2f} GB")
        
        # Initialize sinogram array
        sinogram = cp.zeros((H, self.config.num_projections), dtype=cp.float32)
        
        # Process in batches
        batch_size = self.config.projection_batch_size
        num_batches = (self.config.num_projections + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Loading projection batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, self.config.num_projections)
            
            try:
                batch_data = self._load_batch(proj_folder, start_num, batch_start, batch_end, (H, W))
                sinogram[:, batch_start:batch_end] = batch_data
                
                # Clean up batch memory
                del batch_data
                GPUMemoryManager.cleanup()
                
            except Exception as e:
                logger.error(f"Failed to load batch {batch_idx}: {str(e)}")
                raise
        
        logger.info(f"Loaded sinogram with shape: {sinogram.shape}")
        return sinogram
    
    def _load_batch(self, proj_folder: str, start_num: int, batch_start: int, 
                   batch_end: int, proj_shape: Tuple[int, int]) -> cp.ndarray:
        """Load and process a batch of projections"""
        H, W = proj_shape
        batch_size = batch_end - batch_start
        batch_projections = []
        
        # Load projections
        for i in range(batch_start, batch_end):
            proj_file = os.path.join(proj_folder, f"Projection_{start_num + i:04d}.tiff")
            
            if os.path.exists(proj_file):
                try:
                    proj = np.array(Image.open(proj_file))
                    batch_projections.append(proj)
                except Exception as e:
                    logger.warning(f"Failed to load {proj_file}: {str(e)}")
                    batch_projections.append(np.zeros((H, W), dtype=np.uint16))
            else:
                logger.warning(f"Missing projection: {proj_file}")
                batch_projections.append(np.zeros((H, W), dtype=np.uint16))
        
        # Convert to GPU and preprocess
        batch_gpu = cp.array(np.stack(batch_projections, axis=2))  # Shape: (H, W, batch_size)
        
        # Process each projection
        processed_batch = cp.zeros((H, batch_size), dtype=cp.float32)
        
        for b in range(batch_size):
            proj_gpu = batch_gpu[:, :, b]
            proj_processed = self.preprocessor.preprocess_projection_gpu(proj_gpu)
            
            # Create sinogram by summing along detector rows (simple approach)
            # For proper cone beam reconstruction, this should be more sophisticated
            proj_sinogram = cp.mean(proj_processed, axis=0)  # Average along rows
            processed_batch[:, b] = proj_sinogram[:, np.newaxis].repeat(H, axis=1).mean(axis=1)
        
        return processed_batch

class CBCTFilter:
    """CBCT filtering operations"""
    
    def __init__(self, config: CBCTConfig):
        self.config = config
    
    def filter_projections(self, sinogram: cp.ndarray) -> np.ndarray:
        """Filter sinogram with automatic memory management"""
        sino_np = cp.asnumpy(sinogram) if isinstance(sinogram, cp.ndarray) else sinogram
        
        # Estimate memory requirements
        memory_est = GPUMemoryManager.estimate_sinogram_memory(sino_np.shape)
        memory_info = GPUMemoryManager.get_memory_info()
        
        logger.info(f"Filter memory estimate: {memory_est['total']:.2f} GB")
        logger.info(f"Available GPU memory: {memory_info['device_free']:.2f} GB")
        
        # Choose filtering strategy based on memory
        memory_limit = memory_info['device_free'] * self.config.max_gpu_memory_fraction
        
        if memory_est['total'] <= memory_limit:
            logger.info("Using full GPU filtering")
            return self._filter_gpu_full(sino_np)
        else:
            logger.info("Using chunked GPU filtering")
            return self._filter_gpu_chunked(sino_np)
    
    def _filter_gpu_full(self, sino_np: np.ndarray) -> np.ndarray:
        """Full GPU filtering"""
        try:
            sino_gpu = cp.asarray(sino_np, dtype=cp.float32)
            filtered = self._apply_ramp_filter(sino_gpu)
            result = cp.asnumpy(filtered)
            
            # Cleanup
            del sino_gpu, filtered
            GPUMemoryManager.cleanup()
            
            return result
            
        except cp.cuda.memory.OutOfMemoryError:
            logger.warning("GPU OOM in full filtering, falling back to chunked")
            return self._filter_gpu_chunked(sino_np)
        except Exception as e:
            logger.error(f"GPU filtering failed: {str(e)}")
            return self._filter_cpu_fallback(sino_np)
    
    def _filter_gpu_chunked(self, sino_np: np.ndarray) -> np.ndarray:
        """Chunked GPU filtering"""
        projLen, numAngles = sino_np.shape
        chunk_size = self.config.filter_chunk_size
        
        # Pre-compute filter
        filter_gpu = self._create_ramp_filter(projLen)
        
        # Initialize output
        filtered_sino = np.zeros_like(sino_np, dtype=np.float32)
        
        try:
            num_chunks = (numAngles + chunk_size - 1) // chunk_size
            
            for chunk_idx in tqdm(range(num_chunks), desc="Filtering chunks"):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, numAngles)
                
                # Process chunk
                chunk_np = sino_np[:, start_idx:end_idx]
                chunk_gpu = cp.asarray(chunk_np, dtype=cp.float32)
                
                # Apply filter
                chunk_filtered = self._apply_ramp_filter_with_precomputed(chunk_gpu, filter_gpu)
                filtered_sino[:, start_idx:end_idx] = cp.asnumpy(chunk_filtered)
                
                # Cleanup
                del chunk_gpu, chunk_filtered
                GPUMemoryManager.cleanup()
            
            return filtered_sino
            
        except Exception as e:
            logger.error(f"Chunked filtering failed: {str(e)}")
            return self._filter_cpu_fallback(sino_np)
        finally:
            del filter_gpu
            GPUMemoryManager.cleanup()
    
    def _create_ramp_filter(self, projLen: int) -> cp.ndarray:
        """Create ramp filter on GPU"""
        a = 0.5
        step = 2 * np.pi / projLen
        
        w = cp.arange(-cp.pi, cp.pi, step, dtype=cp.float32)
        if w.size < projLen:
            w = cp.concatenate([w, w[-1:] + step])
        
        rn1 = cp.abs(2 / a * cp.sin(a * w / 2))
        rn2 = cp.sin(a * w / 2) / (a * w / 2)
        rn2 = cp.nan_to_num(rn2, nan=1.0)
        r = rn1 * (rn2 ** 2)
        
        return cp.fft.fftshift(r).astype(cp.complex64)
    
    def _apply_ramp_filter(self, sino_gpu: cp.ndarray) -> cp.ndarray:
        """Apply ramp filter to sinogram"""
        projLen = sino_gpu.shape[0]
        filt = self._create_ramp_filter(projLen)
        
        projfft = cp.fft.fft(sino_gpu, axis=0)
        filtProj = projfft * filt[:, None]
        filtered = cp.real(cp.fft.ifft(filtProj, axis=0)).astype(cp.float32)
        
        return filtered
    
    def _apply_ramp_filter_with_precomputed(self, sino_gpu: cp.ndarray, 
                                          filt: cp.ndarray) -> cp.ndarray:
        """Apply precomputed ramp filter"""
        projfft = cp.fft.fft(sino_gpu, axis=0)
        filtProj = projfft * filt[:, None]
        return cp.real(cp.fft.ifft(filtProj, axis=0)).astype(cp.float32)
    
    def _filter_cpu_fallback(self, sino_np: np.ndarray) -> np.ndarray:
        """CPU fallback filtering"""
        logger.info("Using CPU fallback for filtering")
        
        projLen, numAngles = sino_np.shape
        a = 0.5
        step = 2 * np.pi / projLen
        
        w = np.arange(-np.pi, np.pi, step, dtype=np.float32)
        if w.size < projLen:
            w = np.concatenate([w, w[-1:] + step])
        
        rn1 = np.abs(2 / a * np.sin(a * w / 2))
        rn2 = np.sin(a * w / 2) / (a * w / 2)
        rn2 = np.nan_to_num(rn2, nan=1.0)
        r = rn1 * (rn2 ** 2)
        filt = np.fft.fftshift(r).astype(np.complex64)
        
        projfft = np.fft.fft(sino_np, axis=0)
        filtProj = projfft * filt[:, None]
        return np.real(np.fft.ifft(filtProj, axis=0)).astype(np.float32)

class CBCTBackprojector:
    """CBCT backprojection"""
    
    def __init__(self, config: CBCTConfig):
        self.config = config
        self._compile_backprojection_kernel()
    
    def _compile_backprojection_kernel(self):
        """Compile CUDA backprojection kernel"""
        kernel_source = r"""
        extern "C" __global__
        void backproj_kernel(const float* __restrict__ sino,
                             const float* __restrict__ sin_th,
                             const float* __restrict__ cos_th,
                             float* __restrict__ out,
                             const int N,
                             const int A,
                             const float magnification)
        {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x >= N || y >= N) return;

            float xf = (float)x - 0.5f * (float)N;
            float yf = (float)y - 0.5f * (float)N;

            float acc = 0.0f;
            for (int a = 0; a < A; ++a) {
                float s = xf * sin_th[a] - yf * cos_th[a] + 0.5f * (float)N;
                int si = __float2int_rn(s);
                if (0 <= si && si < N) {
                    acc += sino[si * A + a];
                }
            }
            out[y * N + x] = acc * magnification;
        }
        """
        self.backproj_kernel = cp.RawKernel(kernel_source, "backproj_kernel")
    
    def backproject(self, filtered_sino: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Perform GPU backprojection with cone beam geometry"""
        logger.info("Starting backprojection")
        
        sino_gpu = cp.asarray(filtered_sino, dtype=cp.float32)
        N, A = sino_gpu.shape
        
        # Convert angles for CBCT geometry (starts at 270° clockwise)
        theta_cbct = (angles + self.config.start_angle) % 360
        theta_rad = cp.asarray(theta_cbct, dtype=cp.float32) * (cp.pi / 180.0)
        
        sin_th = cp.sin(theta_rad).astype(cp.float32)
        cos_th = cp.cos(theta_rad).astype(cp.float32)
        
        # Calculate magnification factor
        magnification = self.config.source_detector_dist / self.config.source_object_dist
        logger.info(f"Using magnification factor: {magnification:.2f}")
        
        # Allocate output
        out = cp.zeros((N, N), dtype=cp.float32)
        
        # Launch kernel
        block = (16, 16, 1)
        grid = ((N + block[0] - 1) // block[0], (N + block[1] - 1) // block[1], 1)
        
        self.backproj_kernel(
            grid, block,
            (sino_gpu.ravel(), sin_th, cos_th, out.ravel(),
             np.int32(N), np.int32(A), np.float32(magnification))
        )
        
        # Apply final transformations
        backproj_result = cp.flipud(out)
        result = cp.asnumpy(backproj_result)
        
        # Cleanup
        del sino_gpu, out, backproj_result
        GPUMemoryManager.cleanup()
        
        logger.info(f"Backprojection completed, result shape: {result.shape}")
        return result

class CBCTReconstructor:
    """Main CBCT reconstruction class"""
    
    def __init__(self, config: CBCTConfig):
        self.config = config
        self.data_loader = CBCTDataLoader(config)
        self.filter = CBCTFilter(config)
        self.backprojector = CBCTBackprojector(config)
    
    def reconstruct(self, projection_folder: str, output_folder: str) -> Dict[str, Any]:
        """Perform complete CBCT reconstruction"""
        logger.info("=== Starting CBCT Reconstruction ===")
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize GPU
        self._initialize_gpu()
        
        start_time = time.perf_counter()
        
        try:
            # Step 1: Load projections
            logger.info("--- Loading and Preprocessing Projections ---")
            sinogram = self.data_loader.load_projections(projection_folder)
            
            # Step 2: Filter projections
            logger.info("--- Filtering Projections ---")
            filtered_sino = self.filter.filter_projections(sinogram)
            
            # Step 3: Create angle array
            angles = np.linspace(0, self.config.scan_angle, 
                               self.config.num_projections, endpoint=False)
            
            # Step 4: Backproject
            logger.info("--- Backprojecting ---")
            reconstruction = self.backprojector.backproject(filtered_sino, angles)
            
            # Step 5: Post-processing
            if self.config.volume_denoising_enabled:
                logger.info("--- Volume Denoising ---")
                reconstruction = self._apply_volume_denoising(reconstruction)
            
            end_time = time.perf_counter()
            reconstruction_time = end_time - start_time
            
            logger.info(f"Total reconstruction time: {reconstruction_time:.2f} seconds")
            
            # Step 6: Save results
            results = self._save_results(reconstruction, output_folder, reconstruction_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {str(e)}")
            traceback.print_exc()
            raise
        finally:
            GPUMemoryManager.cleanup()
    
    def _initialize_gpu(self):
        """Initialize GPU and warm up"""
        logger.info("Initializing GPU...")
        
        # Warm up with dummy data
        dummy_sino = cp.ones((256, 180), dtype=cp.float32)
        dummy_angles = cp.linspace(0, 180, 180, dtype=cp.float32)
        
        # Test operations
        _ = cp.fft.fft(dummy_sino[:, :10], axis=0)
        
        memory_info = GPUMemoryManager.get_memory_info()
        logger.info(f"GPU initialized. Available memory: {memory_info['device_free']:.2f} GB")
    
    def _apply_volume_denoising(self, volume: np.ndarray) -> np.ndarray:
        """Apply volume denoising"""
        try:
            from cupyx.scipy.ndimage import gaussian_filter
            
            volume_gpu = cp.asarray(volume)
            
            for i in range(self.config.denoising_iterations):
                volume_gpu = gaussian_filter(volume_gpu, sigma=self.config.denoising_sigma)
            
            result = cp.asnumpy(volume_gpu)
            del volume_gpu
            GPUMemoryManager.cleanup()
            
            return result
            
        except ImportError:
            logger.warning("cupyx not available, skipping volume denoising")
            return volume
    
    def _save_results(self, reconstruction: np.ndarray, output_folder: str, 
                     reconstruction_time: float) -> Dict[str, Any]:
        """Save reconstruction results"""
        logger.info("--- Saving Results ---")
        
        # Normalize to 16-bit
        recon_norm = np.round(
            (reconstruction - np.min(reconstruction)) / np.ptp(reconstruction) * 65535
        )
        recon_16bit = recon_norm.astype(np.uint16)
        
        # Save raw volume
        output_raw_path = os.path.join(output_folder, 'cvolume.raw')
        recon_16bit.tofile(output_raw_path)
        logger.info(f"Saved raw volume: {output_raw_path}")
        
        # Save metadata
        metadata = {
            'shape': reconstruction.shape,
            'voxel_size': self.config.voxel_size,
            'data_type': 'uint16',
            'reconstruction_time': reconstruction_time,
            'config': self.config.__dict__
        }
        
        metadata_path = os.path.join(output_folder, 'reconstruction_metadata.npz')
        np.savez(metadata_path, **metadata)
        logger.info(f"Saved metadata: {metadata_path}")
        
        # Save preview slice
        middle_slice = reconstruction[reconstruction.shape[0]//2, :, :]
        middle_norm = np.round(
            (middle_slice - np.min(middle_slice)) / np.ptp(middle_slice) * 255
        )
        middle_img = Image.fromarray(middle_norm.astype(np.uint8))
        preview_path = os.path.join(output_folder, 'middle_slice_preview.png')
        middle_img.save(preview_path)
        logger.info(f"Saved preview: {preview_path}")
        
        return {
            'reconstruction': reconstruction,
            'shape': reconstruction.shape,
            'reconstruction_time': reconstruction_time,
            'output_folder': output_folder,
            'raw_file': output_raw_path,
            'metadata_file': metadata_path,
            'preview_file': preview_path
        }

def main():
    """Main execution function"""
    
    # Configuration
    config = CBCTConfig(
        num_projections=1600,
        scan_angle=360.0,
        start_angle=270.0,
        projection_batch_size=25,  # Reduced for memory safety
        filter_chunk_size=50,      # Reduced for memory safety
        max_gpu_memory_fraction=0.7  # Conservative memory usage
    )
    
    # Paths
    projection_folder = 'data/cbct_projections/'
    output_folder = 'data/cbct_reconstructed/'
    
    # Check input folder
    if not os.path.exists(projection_folder):
        logger.error(f"Projection folder not found: {projection_folder}")
        return
    
    try:
        # Create reconstructor and run
        reconstructor = CBCTReconstructor(config)
        results = reconstructor.reconstruct(projection_folder, output_folder)
        
        logger.info("=== CBCT Reconstruction Complete! ===")
        logger.info(f"Volume shape: {results['shape']}")
        logger.info(f"Reconstruction time: {results['reconstruction_time']:.2f} seconds")
        logger.info(f"Results saved in: {results['output_folder']}")
        
    except KeyboardInterrupt:
        logger.info("Reconstruction interrupted by user")
    except Exception as e:
        logger.error(f"Reconstruction failed: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())