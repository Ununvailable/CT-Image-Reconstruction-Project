# Refactored CBCT reconstruction script
# Purpose: Accept full 3D projection stacks (H, W, N), perform per-detector-row
# ramp filtering (along detector u / width axis), and keep memory-safe chunking.
# This file consolidates loader, preprocessor, GPU memory manager, filter, 3D backprojection,
# and main orchestration. Adapted from user's original script to avoid dimensionality
# mismatches and to perform correct CBCT filtering.

import numpy as np
import os
import time
import cupy as cp
from PIL import Image
from tqdm import tqdm
import logging
import traceback
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CBCTConfig:
    num_projections: int = 1600
    scan_angle: float = 360.0
    start_angle: float = 270.0
    detector_size: Tuple[int, int] = (2860, 2860)  # (H, W)
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
    projection_batch_size: int = 50  # how many projections to load per batch
    filter_row_chunk: int = 16       # how many detector rows (v) to filter at once
    filter_angle_chunk: int = 64     # how many angles to process together when reshaping
    max_gpu_memory_fraction: float = 0.8


class GPUMemoryManager:
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        try:
            mempool = cp.get_default_memory_pool()
            device_info = cp.cuda.Device().mem_info
            return {
                'used': mempool.used_bytes() / 1e9,
                'pool_total': mempool.total_bytes() / 1e9,
                'device_free': device_info[0] / 1e9,
                'device_total': device_info[1] / 1e9
            }
        except Exception:
            # If GPU query fails, return zeros to force CPU fallback
            return {'used': 0.0, 'pool_total': 0.0, 'device_free': 0.0, 'device_total': 0.0}

    @staticmethod
    def cleanup():
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    @staticmethod
    def estimate_sinogram_memory(shape: Tuple[int, ...]) -> Dict[str, float]:
        """Estimate memory requirements for filtering.

        Accepts either a 2-D sinogram shape (projLen, numAngles) or
        a 3-D projection stack shape (H, W, N).
        Returns memory estimates in GB.
        """
        if len(shape) == 2:
            projLen, numAngles = shape
            num_rows = 1
        elif len(shape) == 3:
            num_rows, projLen, numAngles = shape
        else:
            raise ValueError("Unsupported sinogram/projection shape for memory estimation")

        # float32 for image, complex64 for FFT temporary
        sino_mem = projLen * numAngles * num_rows * 4 / 1e9
        fft_temp = projLen * numAngles * num_rows * 8 / 1e9
        backproj_mem = projLen * projLen * 4 / 1e9  # crude backprojection scratch estimate
        total = sino_mem + fft_temp + backproj_mem

        return {
            'sinogram': sino_mem,
            'fft_temporary': fft_temp,
            'backprojection': backproj_mem,
            'total': total
        }


class CBCTPreprocessor:
    def __init__(self, config: CBCTConfig):
        self.config = config

    def preprocess_projection_gpu(self, proj_gpu: cp.ndarray) -> cp.ndarray:
        try:
            proj_processed = proj_gpu.astype(cp.float32) - self.config.dark_current
            proj_processed = cp.maximum(proj_processed, 1.0)

            if self.config.apply_log_correction:
                I0 = cp.max(proj_processed)
                proj_processed = -cp.log(proj_processed / I0)

            if self.config.apply_bad_pixel_correction:
                proj_processed = self._bad_pixel_correction_gpu(proj_processed)

            if self.config.apply_noise_reduction:
                proj_processed = self._noise_reduction_gpu(proj_processed)

            if self.config.apply_truncation_correction:
                proj_processed = self._truncation_correction_gpu(proj_processed)

            return proj_processed
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def _bad_pixel_correction_gpu(self, proj_gpu: cp.ndarray) -> cp.ndarray:
        try:
            from cupyx.scipy.ndimage import median_filter
            bad_mask = proj_gpu > self.config.bad_pixel_threshold
            if cp.any(bad_mask):
                filtered = median_filter(proj_gpu, size=3)
                return cp.where(bad_mask, filtered, proj_gpu)
            return proj_gpu
        except Exception:
            logger.warning("cupyx not available or failed; skipping bad pixel correction")
            return proj_gpu

    def _noise_reduction_gpu(self, proj_gpu: cp.ndarray) -> cp.ndarray:
        try:
            from cupyx.scipy.ndimage import gaussian_filter
            return gaussian_filter(proj_gpu, sigma=1.5)
        except Exception:
            logger.warning("cupyx not available or failed; skipping noise reduction")
            return proj_gpu

    def _truncation_correction_gpu(self, proj_gpu: cp.ndarray) -> cp.ndarray:
        H, W = proj_gpu.shape
        corrected = proj_gpu.copy()
        width = self.config.truncation_width
        if width > 0 and width < W:
            left_edge = corrected[:, :10].mean(axis=1, keepdims=True)
            corrected[:, :width] = left_edge
            right_edge = corrected[:, -10:].mean(axis=1, keepdims=True)
            corrected[:, -width:] = right_edge
        return corrected


class CBCTDataLoader:
    def __init__(self, config: CBCTConfig):
        self.config = config
        self.preprocessor = CBCTPreprocessor(config)

    def load_projections(self, proj_folder: str, start_num: int = 0) -> cp.ndarray:
        logger.info(f"Loading {self.config.num_projections} CBCT projections from {proj_folder}")

        first_file = os.path.join(proj_folder, f"Projection_{start_num:04d}.tiff")
        if not os.path.exists(first_file):
            raise FileNotFoundError(f"First projection file not found: {first_file}")

        first_proj = np.array(Image.open(first_file))
        H, W = first_proj.shape
        logger.info(f"Projection size: {H} x {W}")

        projection_memory = H * W * self.config.num_projections * 4 / 1e9
        memory_info = GPUMemoryManager.get_memory_info()
        logger.info(f"Estimated projection memory: {projection_memory:.2f} GB")
        logger.info(f"Available GPU memory: {memory_info['device_free']:.2f} GB")

        # Allocate projections array on GPU in (H, W, N) format
        projections = cp.zeros((H, W, self.config.num_projections), dtype=cp.float32)

        batch_size = self.config.projection_batch_size
        num_batches = (self.config.num_projections + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Loading projection batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, self.config.num_projections)
            try:
                batch_data = self._load_batch(proj_folder, start_num, batch_start, batch_end, (H, W))
                projections[:, :, batch_start:batch_end] = batch_data
                del batch_data
                GPUMemoryManager.cleanup()
            except Exception as e:
                logger.error(f"Failed to load batch {batch_idx}: {e}")
                raise

        logger.info(f"Loaded projections with shape: {projections.shape}")
        return projections

    def _load_batch(self, proj_folder: str, start_num: int, batch_start: int,
                    batch_end: int, proj_shape: Tuple[int, int]) -> cp.ndarray:
        H, W = proj_shape
        batch_projections = []
        for i in range(batch_start, batch_end):
            proj_file = os.path.join(proj_folder, f"Projection_{start_num + i:04d}.tiff")
            if os.path.exists(proj_file):
                try:
                    proj = np.array(Image.open(proj_file))
                    batch_projections.append(proj)
                except Exception as e:
                    logger.warning(f"Failed to load {proj_file}: {e}")
                    batch_projections.append(np.zeros((H, W), dtype=np.uint16))
            else:
                logger.warning(f"Missing projection: {proj_file}")
                batch_projections.append(np.zeros((H, W), dtype=np.uint16))

        # Stack along angle axis -> shape (H, W, batch_size)
        batch_np = np.stack(batch_projections, axis=2)
        # Move to GPU and preprocess per-projection
        batch_gpu = cp.array(batch_np, dtype=cp.float32)
        processed_batch = cp.zeros_like(batch_gpu)
        for b in range(batch_gpu.shape[2]):
            processed_batch[:, :, b] = self.preprocessor.preprocess_projection_gpu(batch_gpu[:, :, b])
        return processed_batch


class CBCTFilter:
    """Filter 3D projection stacks by applying 1D ramp along detector u (width) per detector row."""

    def __init__(self, config: CBCTConfig):
        self.config = config

    def filter_projections(self, projections: cp.ndarray) -> cp.ndarray:
        """Entry point: projections shape expected (H, W, N) where
        - H: detector rows (v)
        - W: detector columns (u) — ramp filter axis
        - N: projection angles
        Returns filtered stack with same shape.
        """
        # Ensure numpy/cupy consistent type for memory estimation
        proj_shape = projections.shape
        logger.info(f"Filtering projections of shape: {proj_shape}")

        # Estimate memory for the worst-case chunk (use 3D estimator)
        mem_est = GPUMemoryManager.estimate_sinogram_memory(proj_shape)
        mem_info = GPUMemoryManager.get_memory_info()
        logger.info(f"Filter memory estimate: {mem_est['total']:.2f} GB")
        logger.info(f"Available GPU memory: {mem_info['device_free']:.2f} GB")

        # Prefer GPU processing when some free memory exists
        try_gpu = mem_info['device_free'] > 0.1

        # Process in row-chunks to keep memory bounded
        H, W, N = proj_shape
        row_chunk = max(1, min(self.config.filter_row_chunk, H))

        filtered = cp.zeros_like(projections, dtype=cp.float32)

        # Precompute frequency-domain filter on GPU for length W
        filt_gpu = self._create_ramp_filter_gpu(W)

        for v_start in tqdm(range(0, H, row_chunk), desc='Filtering rows'):
            v_end = min(v_start + row_chunk, H)
            # slice shape (rows, W, N)
            slice_gpu = projections[v_start:v_end, :, :]

            try:
                # reshape to (W, rows*N) to FFT along axis=0 (detector u)
                rows = v_end - v_start
                reshaped = slice_gpu.transpose(1, 0, 2).reshape(W, rows * N)

                if try_gpu:
                    reshaped_gpu = cp.asarray(reshaped, dtype=cp.float32)
                    # FFT along axis 0
                    projfft = cp.fft.fft(reshaped_gpu, axis=0)
                    filtered_fft = projfft * filt_gpu[:, None]
                    filtered_ifft = cp.real(cp.fft.ifft(filtered_fft, axis=0)).astype(cp.float32)
                    # reshape back to (rows, W, N)
                    back = filtered_ifft.reshape(W, rows, N).transpose(1, 0, 2)
                    filtered[v_start:v_end, :, :] = back
                    del reshaped_gpu, projfft, filtered_fft, filtered_ifft, back
                    GPUMemoryManager.cleanup()
                else:
                    # CPU fallback using numpy
                    reshaped_cpu = reshaped.get() if isinstance(reshaped, cp.ndarray) else reshaped
                    projfft = np.fft.fft(reshaped_cpu, axis=0)
                    filtered_fft = projfft * cp.asnumpy(filt_gpu)[:, None]
                    filtered_ifft = np.real(np.fft.ifft(filtered_fft, axis=0)).astype(np.float32)
                    back = filtered_ifft.reshape(W, rows, N).transpose(1, 0, 2)
                    filtered[v_start:v_end, :, :] = cp.array(back)
                    del reshaped_cpu, projfft, filtered_fft, filtered_ifft, back
                    GPUMemoryManager.cleanup()

            except cp.cuda.memory.OutOfMemoryError:
                logger.warning("OOM while filtering a row block; reducing chunk size and retrying")
                # reduce row_chunk and retry for this block
                if row_chunk == 1:
                    logger.error("Cannot reduce row chunk further; aborting filter")
                    raise
                # halve chunk size and retry loop by re-queueing range
                self.config.filter_row_chunk = max(1, row_chunk // 2)
                row_chunk = self.config.filter_row_chunk
                continue
            except Exception as e:
                logger.error(f"Filtering failed for rows {v_start}:{v_end} -> {e}")
                traceback.print_exc()
                raise

        # return as GPU array (same layout)
        GPUMemoryManager.cleanup()
        return filtered

    def _create_ramp_filter_gpu(self, projLen: int) -> cp.ndarray:
        # construct frequency response (complex) on GPU
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


class CBCTBackprojector3D:
    def __init__(self, config: CBCTConfig):
        self.config = config
        self._compile_3d_backprojection_kernel()

    def _compile_3d_backprojection_kernel(self):
        kernel_source = r"""
        extern "C" __global__
        void backproj_3d_kernel(const float* __restrict__ projections,
                                const float* __restrict__ angles,
                                float* __restrict__ volume,
                                const int proj_h, const int proj_w, const int num_angles,
                                const int vol_x, const int vol_y, const int vol_z,
                                const float source_origin, const float origin_detector,
                                const float pixel_size, const float voxel_size)
        {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            int z = blockDim.z * blockIdx.z + threadIdx.z;
            if (x >= vol_x || y >= vol_y || z >= vol_z) return;

            float world_x = (x - vol_x/2.0f) * voxel_size;
            float world_y = (y - vol_y/2.0f) * voxel_size;
            float world_z = (z - vol_z/2.0f) * voxel_size;

            float acc = 0.0f;
            for (int angle_idx = 0; angle_idx < num_angles; angle_idx++) {
                float angle = angles[angle_idx];
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float source_x = source_origin * sin_a;
                float source_y = -source_origin * cos_a;
                float ray_x = world_x - source_x;
                float ray_y = world_y - source_y;
                float t = (source_origin + origin_detector) / (source_origin + ray_y * cos_a + ray_x * sin_a);
                float det_u = (ray_x * cos_a - ray_y * sin_a) * t / pixel_size + proj_w / 2.0f;
                float det_v = world_z * t / pixel_size + proj_h / 2.0f;
                int u0 = (int)floorf(det_u);
                int v0 = (int)floorf(det_v);
                int u1 = u0 + 1;
                int v1 = v0 + 1;
                if (u0 >= 0 && u1 < proj_w && v0 >= 0 && v1 < proj_h) {
                    float wu = det_u - u0;
                    float wv = det_v - v0;
                    int base_idx = angle_idx * proj_h * proj_w;
                    float val = (1-wu) * (1-wv) * projections[base_idx + v0 * proj_w + u0] +
                               wu * (1-wv) * projections[base_idx + v0 * proj_w + u1] +
                               (1-wu) * wv * projections[base_idx + v1 * proj_w + u0] +
                               wu * wv * projections[base_idx + v1 * proj_w + u1];
                    acc += val / (t * t);
                }
            }
            volume[z * vol_x * vol_y + y * vol_x + x] = acc * 3.14159f / num_angles;
        }
        """
        self.backproj_3d_kernel = cp.RawKernel(kernel_source, "backproj_3d_kernel")

    def backproject(self, projections: cp.ndarray, angles: np.ndarray) -> np.ndarray:
        logger.info("Starting 3D cone beam backprojection")
        proj_h, proj_w, num_angles = projections.shape
        vol_size = min(proj_h, proj_w) // 4
        vol_x = vol_y = vol_z = vol_size
        logger.info(f"Volume size: {vol_x} x {vol_y} x {vol_z}")

        # Convert angles to radians on GPU
        theta_cbct = (angles + self.config.start_angle) % 360
        theta_rad = cp.asarray(theta_cbct * (np.pi / 180.0), dtype=cp.float32)

        volume = cp.zeros((vol_z, vol_y, vol_x), dtype=cp.float32)
        proj_reshaped = cp.transpose(projections, (2, 0, 1))  # (angles, h, w)

        block = (8, 8, 8)
        grid = ((vol_x + block[0] - 1) // block[0],
                (vol_y + block[1] - 1) // block[1],
                (vol_z + block[2] - 1) // block[2])

        # flatten arrays for kernel
        try:
            self.backproj_3d_kernel(
                grid, block,
                (proj_reshaped.ravel(), theta_rad, volume.ravel(),
                 np.int32(proj_h), np.int32(proj_w), np.int32(num_angles),
                 np.int32(vol_x), np.int32(vol_y), np.int32(vol_z),
                 np.float32(self.config.source_object_dist),
                 np.float32(self.config.source_detector_dist - self.config.source_object_dist),
                 np.float32(self.config.pixel_size),
                 np.float32(self.config.voxel_size))
            )
        except Exception as e:
            logger.error(f"Backprojection kernel launch failed: {e}")
            raise

        result = cp.asnumpy(volume)
        del projections, volume, proj_reshaped
        GPUMemoryManager.cleanup()
        logger.info(f"3D backprojection completed, result shape: {result.shape}")
        return result


class CBCTReconstructor:
    def __init__(self, config: CBCTConfig):
        self.config = config
        self.data_loader = CBCTDataLoader(config)
        self.filter = CBCTFilter(config)
        self.backprojector = CBCTBackprojector3D(config)

    def reconstruct(self, projection_folder: str, output_folder: str) -> Dict[str, Any]:
        logger.info("=== Starting CBCT Reconstruction ===")
        os.makedirs(output_folder, exist_ok=True)
        self._initialize_gpu()
        start_time = time.perf_counter()

        try:
            logger.info("--- Loading and Preprocessing Projections ---")
            projections = self.data_loader.load_projections(projection_folder)

            logger.info("--- Filtering Projections (per detector row) ---")
            filtered = self.filter.filter_projections(projections)

            logger.info("--- Creating angle array ---")
            angles = np.linspace(0, self.config.scan_angle, self.config.num_projections, endpoint=False)

            logger.info("--- Backprojecting ---")
            reconstruction = self.backprojector.backproject(filtered, angles)

            if self.config.volume_denoising_enabled:
                logger.info("--- Volume Denoising ---")
                reconstruction = self._apply_volume_denoising(reconstruction)

            end_time = time.perf_counter()
            reconstruction_time = end_time - start_time
            logger.info(f"Total reconstruction time: {reconstruction_time:.2f} seconds")

            results = self._save_results(reconstruction, output_folder, reconstruction_time)
            return results

        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            traceback.print_exc()
            raise
        finally:
            GPUMemoryManager.cleanup()

    def _initialize_gpu(self):
        logger.info("Initializing GPU...")
        try:
            dummy_sino = cp.ones((256, 180), dtype=cp.float32)
            _ = cp.fft.fft(dummy_sino[:, :10], axis=0)
            mem = GPUMemoryManager.get_memory_info()
            logger.info(f"GPU initialized. Available memory: {mem['device_free']:.2f} GB")
        except Exception as e:
            logger.warning(f"GPU initialization warning: {e}")

    def _apply_volume_denoising(self, volume: np.ndarray) -> np.ndarray:
        try:
            from cupyx.scipy.ndimage import gaussian_filter
            volume_gpu = cp.asarray(volume)
            for _ in range(self.config.denoising_iterations):
                volume_gpu = gaussian_filter(volume_gpu, sigma=self.config.denoising_sigma)
            result = cp.asnumpy(volume_gpu)
            del volume_gpu
            GPUMemoryManager.cleanup()
            return result
        except Exception:
            logger.warning("cupyx not available; skipping volume denoising")
            return volume

    def _save_results(self, reconstruction: np.ndarray, output_folder: str, reconstruction_time: float) -> Dict[str, Any]:
        logger.info("--- Saving Results ---")
        recon_norm = np.round((reconstruction - np.min(reconstruction)) / np.ptp(reconstruction) * 65535)
        recon_16bit = recon_norm.astype(np.uint16)
        output_raw_path = os.path.join(output_folder, 'cvolume.raw')
        recon_16bit.tofile(output_raw_path)
        logger.info(f"Saved raw volume: {output_raw_path}")

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

        if len(reconstruction.shape) == 3:
            middle_slice = reconstruction[reconstruction.shape[0] // 2, :, :]
        else:
            middle_slice = reconstruction
        middle_norm = np.round((middle_slice - np.min(middle_slice)) / np.ptp(middle_slice) * 255)
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
    config = CBCTConfig(
        num_projections=100,
        scan_angle=360.0,
        start_angle=270.0,
        projection_batch_size=10,
        filter_row_chunk=32,
        filter_angle_chunk=64,
        max_gpu_memory_fraction=0.7
    )

    projection_folder = 'data/cbct_projections/'
    output_folder = 'data/cbct_reconstructed/'

    if not os.path.exists(projection_folder):
        logger.error(f"Projection folder not found: {projection_folder}")
        return

    try:
        reconstructor = CBCTReconstructor(config)
        results = reconstructor.reconstruct(projection_folder, output_folder)
        logger.info("=== CBCT Reconstruction Complete! ===")
        logger.info(f"Volume shape: {results['shape']}")
        logger.info(f"Reconstruction time: {results['reconstruction_time']:.2f} seconds")
        logger.info(f"Results saved in: {results['output_folder']}")
    except KeyboardInterrupt:
        logger.info("Reconstruction interrupted by user")
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
