import numpy as np
import os
import glob
from pathlib import Path
from PIL import Image
import time
import cupy as cp
from tqdm import tqdm

def preprocess_projection_gpu(proj_gpu, dark_current=100, apply_log=True):
    """
    GPU preprocessing for CBCT projections
    Args:
        proj_gpu: CuPy array of projection
        dark_current: Dark current offset
        apply_log: Apply logarithmic transformation (I0Log)
    """
    # Convert to float32 and apply dark current correction
    proj_processed = proj_gpu.astype(cp.float32) - dark_current
    
    # Clamp to avoid negative values
    proj_processed = cp.maximum(proj_processed, 1.0)
    
    if apply_log:
        # Logarithmic correction: -log(I/I0)
        # Assuming max intensity represents I0
        I0 = cp.max(proj_processed)
        proj_processed = -cp.log(proj_processed / I0)
    
    return proj_processed

def bad_pixel_correction_gpu(proj_gpu, kernel_size=3, threshold=32768):
    """GPU bad pixel correction using median filter"""
    # Simple implementation - replace with more sophisticated if needed
    from cupyx.scipy.ndimage import median_filter
    
    # Detect bad pixels (values above threshold)
    bad_mask = proj_gpu > threshold
    
    if cp.any(bad_mask):
        # Apply median filter only to bad pixels
        filtered = median_filter(proj_gpu, size=kernel_size)
        proj_corrected = cp.where(bad_mask, filtered, proj_gpu)
        return proj_corrected
    
    return proj_gpu

def bilateral_filter_gpu_simple(proj_gpu, spatial_sigma=1.5, range_sigma=1.5):
    """
    Simplified bilateral filter on GPU
    For full implementation, consider using cupyx.scipy.ndimage or custom CUDA kernel
    """
    from cupyx.scipy.ndimage import gaussian_filter
    
    # Simplified version - just Gaussian smoothing
    # Real bilateral filter would need custom implementation
    return gaussian_filter(proj_gpu, sigma=spatial_sigma)

def truncation_correction_gpu(proj_gpu, width_left=500, width_right=500):
    """Apply truncation correction by extending edges"""
    H, W = proj_gpu.shape
    
    # Simple extrapolation - extend edge values
    corrected = proj_gpu.copy()
    
    # Left truncation correction
    if width_left > 0:
        left_edge = corrected[:, :10].mean(axis=1, keepdims=True)
        corrected[:, :width_left] = left_edge
    
    # Right truncation correction
    if width_right > 0:
        right_edge = corrected[:, -10:].mean(axis=1, keepdims=True)
        corrected[:, -width_right:] = right_edge
    
    return corrected

def load_cbct_projections_gpu(proj_folder, num_projections=1600, start_num=0, 
                             batch_size=100, apply_preprocessing=True):
    """
    Load and preprocess CBCT projections with GPU acceleration
    Process in batches to manage memory
    """
    print(f"Loading {num_projections} CBCT projections...")
    
    # Get first projection to determine size
    first_file = os.path.join(proj_folder, f"Projection_{start_num:04d}.tiff")
    if not os.path.exists(first_file):
        raise FileNotFoundError(f"First projection file not found: {first_file}")
    
    first_proj = np.array(Image.open(first_file))
    H, W = first_proj.shape
    print(f"Projection size: {H} x {W}")
    
    # Initialize sinogram array
    sinogram = cp.zeros((H, num_projections), dtype=cp.float32)
    
    # Process in batches
    for batch_start in tqdm(range(0, num_projections, batch_size), 
                           desc="Loading projections"):
        batch_end = min(batch_start + batch_size, num_projections)
        batch_proj = []
        
        # Load batch
        for i in range(batch_start, batch_end):
            proj_file = os.path.join(proj_folder, f"Projection_{start_num + i:04d}.tiff")
            if os.path.exists(proj_file):
                proj = np.array(Image.open(proj_file))
                batch_proj.append(proj)
            else:
                print(f"Warning: Missing projection {proj_file}")
                # Use zeros for missing projection
                batch_proj.append(np.zeros((H, W), dtype=np.uint16))
        
        # Convert batch to GPU and preprocess
        batch_gpu = cp.array(np.stack(batch_proj, axis=2))  # Shape: (H, W, batch_size)
        
        if apply_preprocessing:
            for b in range(batch_gpu.shape[2]):
                proj_gpu = batch_gpu[:, :, b]
                
                # Apply preprocessing steps
                proj_gpu = preprocess_projection_gpu(proj_gpu, apply_log=True)
                proj_gpu = bad_pixel_correction_gpu(proj_gpu, threshold=32768)
                proj_gpu = bilateral_filter_gpu_simple(proj_gpu, 
                                                     spatial_sigma=1.5, 
                                                     range_sigma=1.5)
                proj_gpu = truncation_correction_gpu(proj_gpu, 
                                                   width_left=500, 
                                                   width_right=500)
                
                batch_gpu[:, :, b] = proj_gpu
        
        # Sum along detector rows to create sinogram
        # For cone beam, we might want more sophisticated sinogram creation
        batch_sino = cp.sum(batch_gpu, axis=0)  # Sum along rows (axis=0)
        sinogram[:, batch_start:batch_end] = batch_sino
        
        # Clear batch from GPU memory
        del batch_gpu, batch_proj
        cp.get_default_memory_pool().free_all_blocks()
    
    return sinogram

def cone_beam_geometry_correction(sinogram, 
                                source_object_dist=28.625365287711134,
                                source_detector_dist=699.9996522369905,
                                detector_offset_u=1430.1098329145173,
                                detector_offset_v=1429.4998776624227):
    """
    Apply cone beam geometry corrections
    This is a simplified version - full implementation would need more sophisticated handling
    """
    # Calculate magnification factor
    magnification = source_detector_dist / source_object_dist
    print(f"Magnification factor: {magnification:.2f}")
    
    # Apply detector offset correction (simplified)
    # In practice, this would involve more complex geometric transformations
    H, W = sinogram.shape
    
    # Create coordinate grids for geometric correction
    # This is where you'd implement the full cone beam geometry
    # For now, apply a simple offset correction
    
    return sinogram

# Keep existing GPU functions
def projFilter_gpu(sino_np: np.ndarray) -> np.ndarray:
    """GPU version of projFilter using CuPy FFT along the projection axis (axis=0)."""
    sino = cp.asarray(sino_np, dtype=cp.float32)
    projLen, numAngles = sino.shape
    a = 0.5
    step = 2 * np.pi / projLen

    w = cp.arange(-cp.pi, cp.pi, step, dtype=cp.float32)
    if w.size < projLen:
        w = cp.concatenate([w, w[-1:] + step])

    rn1 = cp.abs(2 / a * cp.sin(a * w / 2))
    rn2 = cp.sin(a * w / 2) / (a * w / 2)
    rn2 = cp.nan_to_num(rn2, nan=1.0)
    r = rn1 * (rn2 ** 2)
    filt = cp.fft.fftshift(r).astype(cp.complex64)

    projfft = cp.fft.fft(sino, axis=0)
    filtProj = projfft * filt[:, None]
    filtSino = cp.real(cp.fft.ifft(filtProj, axis=0)).astype(cp.float32)

    return cp.asnumpy(filtSino)

_backproj_src = r"""
extern "C" __global__
void backproj_kernel(const float* __restrict__ sino,
                     const float* __restrict__ sin_th,
                     const float* __restrict__ cos_th,
                     float* __restrict__ out,
                     const int N,
                     const int A)
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
    out[y * N + x] = acc;
}
""";

_backproj_kernel = cp.RawKernel(_backproj_src, "backproj_kernel")

def backProj_gpu_conebeam(sinogram_np: np.ndarray, theta_deg_np: np.ndarray,
                         geometry_params=None) -> np.ndarray:
    """
    GPU backprojection with cone beam geometry considerations
    """
    sino = cp.asarray(sinogram_np, dtype=cp.float32)
    N, A = sino.shape

    # Convert angles: CBCT starts at 270° clockwise
    theta_cbct = (theta_deg_np + 270) % 360
    theta = cp.asarray(theta_cbct, dtype=cp.float32) * (cp.pi / 180.0)
    
    sin_th = cp.sin(theta).astype(cp.float32)
    cos_th = cp.cos(theta).astype(cp.float32)

    out = cp.zeros((N, N), dtype=cp.float32)

    block = (16, 16, 1)
    grid = ((N + block[0] - 1) // block[0], (N + block[1] - 1) // block[1], 1)

    _backproj_kernel(grid, block, (sino.ravel(), sin_th, cos_th, out.ravel(),
                      np.int32(N), np.int32(A)))

    backprojArray = cp.flipud(out)
    
    # Apply cone beam scaling if geometry params provided
    if geometry_params:
        magnification = geometry_params.get('magnification', 1.0)
        backprojArray = backprojArray * magnification
    
    return cp.asnumpy(backprojArray)

def volume_denoising_gpu(volume_gpu, iterations=4, p1=0.7, p2=0.03):
    """
    3D volume denoising on GPU (simplified implementation)
    Full implementation would use more sophisticated algorithms
    """
    from cupyx.scipy.ndimage import gaussian_filter
    
    # Simple iterative Gaussian filtering
    denoised = volume_gpu.copy()
    for i in range(iterations):
        denoised = gaussian_filter(denoised, sigma=p1)
    
    return denoised

if __name__ == '__main__':
    # CBCT Configuration from ceraAa.config
    cbct_config = {
        'num_projections': 1600,
        'scan_angle': 360,
        'start_angle': 270,  # CBCT starts at 270°
        'detector_size': (2860, 2860),
        'pixel_size': 0.15,  # mm
        'voxel_size': 0.006134010138512811,  # mm
        'source_object_dist': 28.625365287711134,  # mm
        'source_detector_dist': 699.9996522369905,  # mm
        'detector_offset_u': 1430.1098329145173,
        'detector_offset_v': 1429.4998776624227,
        'preprocessing': {
            'i0_log': True,
            'bad_pixel_correction': True,
            'noise_reduction': True,
            'truncation_correction': True,
            'truncation_width': 500
        },
        'volume_denoising': {
            'enabled': True,
            'iterations': 4,
            'p1': 0.7,
            'p2': 0.03
        }
    }
    
    # Define paths
    projection_folder = 'data/cbct_projections/'  # Folder containing Projection_XXXX.tiff
    output_folder = 'data/cbct_reconstructed/'
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print("=== CBCT Reconstruction with CUDA ===")
    
    # GPU initialization
    print("Initializing GPU...")
    dummy_sino = cp.ones((256, 180), dtype=cp.float32)
    dummy_theta = cp.linspace(0, 180, 180, dtype=cp.float32)
    
    # Warm up
    _ = projFilter_gpu(cp.asnumpy(dummy_sino))
    _ = backProj_gpu_conebeam(cp.asnumpy(dummy_sino), cp.asnumpy(dummy_theta))
    print("GPU initialization complete.")
    
    try:
        start_time = time.perf_counter()
        
        # Step 1: Load and preprocess CBCT projections
        print("\n--- Loading CBCT Projections ---")
        sinogram = load_cbct_projections_gpu(
            projection_folder, 
            num_projections=cbct_config['num_projections'],
            start_num=0,
            batch_size=50,  # Adjust based on GPU memory
            apply_preprocessing=True
        )
        
        print(f"Sinogram shape: {sinogram.shape}")
        
        # Step 2: Apply geometry corrections
        print("\n--- Applying Geometry Corrections ---")
        sinogram_corrected = cone_beam_geometry_correction(
            sinogram,
            source_object_dist=cbct_config['source_object_dist'],
            source_detector_dist=cbct_config['source_detector_dist'],
            detector_offset_u=cbct_config['detector_offset_u'],
            detector_offset_v=cbct_config['detector_offset_v']
        )
        
        # Convert to numpy for existing functions
        sinogram_np = cp.asnumpy(sinogram_corrected)
        
        # Step 3: Create angle array
        angles = np.linspace(0, cbct_config['scan_angle'], 
                           cbct_config['num_projections'], endpoint=False)
        
        print(f"\n--- Reconstructing Volume ---")
        print(f"Angles: {len(angles)} projections from 0° to {cbct_config['scan_angle']}°")
        
        # Step 4: Filter projections
        print("Filtering sinogram...")
        filtered_sino = projFilter_gpu(sinogram_np)
        
        # Step 5: Backprojection
        print("Backprojecting...")
        geometry_params = {
            'magnification': cbct_config['source_detector_dist'] / cbct_config['source_object_dist']
        }
        
        recon = backProj_gpu_conebeam(filtered_sino, angles, geometry_params)
        
        # Step 6: Volume denoising (if enabled)
        if cbct_config['volume_denoising']['enabled']:
            print("Applying volume denoising...")
            recon_gpu = cp.asarray(recon)
            recon_denoised = volume_denoising_gpu(
                recon_gpu,
                iterations=cbct_config['volume_denoising']['iterations'],
                p1=cbct_config['volume_denoising']['p1'],
                p2=cbct_config['volume_denoising']['p2']
            )
            recon = cp.asnumpy(recon_denoised)
        
        end_time = time.perf_counter()
        print(f"Total reconstruction time: {end_time - start_time:.2f} seconds")
        
        # Step 7: Save results
        print("\n--- Saving Results ---")
        
        # Normalize and convert to 16-bit
        recon_norm = np.round((recon - np.min(recon)) / np.ptp(recon) * 65535)
        recon_16bit = recon_norm.astype(np.uint16)
        
        # Save as raw volume (as specified in config)
        output_raw_path = os.path.join(output_folder, 'cvolume.raw')
        recon_16bit.tofile(output_raw_path)
        print(f"Saved raw volume: {output_raw_path}")
        
        # Save metadata
        metadata = {
            'shape': recon.shape,
            'voxel_size': cbct_config['voxel_size'],
            'data_type': 'uint16',
            'reconstruction_time': end_time - start_time,
            'config': cbct_config
        }
        
        metadata_path = os.path.join(output_folder, 'reconstruction_metadata.npz')
        np.savez(metadata_path, **metadata)
        print(f"Saved metadata: {metadata_path}")
        
        # Save middle slice as preview
        middle_slice = recon[recon.shape[0]//2, :, :]
        middle_norm = np.round((middle_slice - np.min(middle_slice)) / np.ptp(middle_slice) * 255)
        middle_img = Image.fromarray(middle_norm.astype(np.uint8))
        preview_path = os.path.join(output_folder, 'middle_slice_preview.png')
        middle_img.save(preview_path)
        print(f"Saved preview: {preview_path}")
        
        print(f"\n=== CBCT Reconstruction Complete! ===")
        print(f"Volume shape: {recon.shape}")
        print(f"Voxel size: {cbct_config['voxel_size']} mm")
        print(f"Results saved in: {output_folder}")
        
    except Exception as e:
        print(f"Error during reconstruction: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()