#!/usr/bin/env python3
"""
CBCT 3D Shepp-Logan Phantom Dataset Generation
Generates cone-beam CT projections matching real microCT specifications
Compatible with ASTRA reconstruction pipeline
"""

import numpy as np
import astra
import os
import json
import logging
from tqdm import tqdm
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_3d_shepplogan_phantom(size=512):
    """
    Create 3D Shepp-Logan phantom
    Based on standard ellipsoid parameters
    """
    logger.info(f"Generating {size}³ Shepp-Logan phantom...")
    
    phantom = np.zeros((size, size, size), dtype=np.float32)
    
    # Shepp-Logan ellipsoid parameters: [intensity, a, b, c, x0, y0, z0, phi, theta, psi]
    ellipsoids = [
        [1.0,   0.69,  0.92,  0.90,  0.0,   0.0,    0.0,     0,  0,  0],
        [-1.0,  0.6624, 0.874, 0.880, 0.0,  -0.0184, 0.0,     0,  0,  0],
        [-0.2,  0.11,  0.31,  0.22,  0.22,  0.0,   0.0,    -18,  0, 10],
        [-0.2,  0.16,  0.41,  0.28, -0.22,  0.0,   0.0,     18,  0, 10],
        [0.1,   0.21,  0.25,  0.41,  0.0,   0.35,  -0.15,    0,  0,  0],
        [0.1,   0.046, 0.046, 0.05,  0.0,   0.1,   0.25,     0,  0,  0],
        [0.1,   0.046, 0.046, 0.05,  0.0,  -0.1,   0.25,     0,  0,  0],
        [0.1,   0.046, 0.023, 0.05, -0.08, -0.605, 0.0,      0,  0,  0],
        [0.1,   0.023, 0.023, 0.02,  0.0,  -0.606, 0.0,      0,  0,  0],
        [0.1,   0.023, 0.046, 0.02,  0.06, -0.605, 0.0,      0,  0,  0],
    ]

    
    z, y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j, -1:1:size*1j]
    
    for params in ellipsoids:
        intensity, a, b, c, x0, y0, z0, phi, theta, psi = params
        
        # Rotation matrices (simplified - no rotation for basic phantom)
        phi_rad = np.radians(phi)
        theta_rad = np.radians(theta)
        psi_rad = np.radians(psi)
        
        # Translate coordinates
        xr = x - x0
        yr = y - y0
        zr = z - z0
        
        # Apply rotations if needed
        if phi != 0 or theta != 0 or psi != 0:
            # Simplified rotation around z-axis
            cos_phi = np.cos(phi_rad)
            sin_phi = np.sin(phi_rad)
            x_rot = xr * cos_phi + yr * sin_phi
            y_rot = -xr * sin_phi + yr * cos_phi
            xr, yr = x_rot, y_rot
        
        # Ellipsoid equation
        ellipsoid = (xr/a)**2 + (yr/b)**2 + (zr/c)**2 <= 1
        phantom += intensity * ellipsoid
    
    # After all ellipsoids are added
    phantom = phantom - phantom.min()
    phantom = phantom / phantom.max()  # Normalize to [0, 1]
    phantom = phantom ** 0.5           # Gamma correction to enhance mid-tones
    phantom = phantom * 50.0  # Scale up for better contrast
    # phantom = np.maximum(phantom, 0)  # Ensure non-negative
    logger.info(f"Phantom range: [{phantom.min():.3f}, {phantom.max():.3f}]")

    return phantom


def generate_cbct_projections(phantom, config):
    """
    Generate cone-beam CT projections using ASTRA
    """
    logger.info("Setting up ASTRA cone-beam geometry...")
    
    vol_size = phantom.shape[0]
    
    # Volume geometry (physical size in mm)
    vol_size_mm = config["volume_size_mm"]
    vol_geom = astra.create_vol_geom(
        vol_size, vol_size, vol_size,
        -vol_size_mm[0]/2, vol_size_mm[0]/2,
        -vol_size_mm[1]/2, vol_size_mm[1]/2,
        -vol_size_mm[2]/2, vol_size_mm[2]/2
    )
    
    # Create ASTRA volume
    phantom_id = astra.data3d.create('-vol', vol_geom)
    astra.data3d.store(phantom_id, phantom)
    
    # Projection geometry
    num_angles = config["num_projections"]
    det_rows, det_cols = config["detector_pixels"]
    
    angles = np.linspace(
        np.radians(config["start_angle"]),
        np.radians(config["start_angle"] + config["scan_angle_degrees"]),
        num_angles,
        endpoint=False
    )
    
    # Calculate pixel sizes
    det_size_mm = config["detector_size_mm"]
    pixel_size_u = det_size_mm[0] / det_cols
    pixel_size_v = det_size_mm[1] / det_rows
    
    # Detector offsets (centered for synthetic data)
    det_offset_u = config.get("detector_offset_u", 0.0)
    det_offset_v = config.get("detector_offset_v", 0.0)
    
    # Source and detector distances
    source_origin = config["source_origin_dist"]
    source_detector = config["source_detector_dist"]
    origin_detector = source_detector - source_origin
    
    proj_geom = astra.create_proj_geom(
        'cone',
        pixel_size_v, pixel_size_u,
        det_rows, det_cols,
        angles,
        source_origin,
        origin_detector,
        det_offset_u,
        det_offset_v
    )
    
    # Create projection data
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    
    # Forward projection
    logger.info("Computing forward projection...")
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['VolumeDataId'] = phantom_id
    cfg['ProjectionDataId'] = proj_id
    
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    # Get projections
    projections = astra.data3d.get(proj_id)
    
    # Apply Beer-Lambert transform for CT reconstruction
    # Convert line integrals to transmission projections
    I0 = np.max(projections)  # or use a fixed reference value
    projections = -np.log((I0 - projections) / I0 + 1e-6)  # Avoid log(0)

    # ASTRA cone-beam quirk: returns (det_rows, angles, det_cols) instead of (angles, det_rows, det_cols)
    # Transpose to correct format for saving individual projection files
    projections = projections.transpose(1, 0, 2)
    
    # Cleanup ASTRA objects
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(phantom_id)
    
    logger.info(f"Projections shape: {projections.shape} (angles, rows, cols)")
    logger.info(f"Projections range: [{projections.min():.6f}, {projections.max():.6f}]")
    
    return projections


def save_projections(projections, output_folder, file_format="tiff"):
    """
    Save projections to disk
    Expects projections in shape (angles, rows, cols) from ASTRA
    Saves one file per angle, each file is (rows, cols) = detector (Height, Width)
    """
    slices_folder = os.path.join(output_folder, "slices")
    os.makedirs(slices_folder, exist_ok=True)
    
    num_projections = projections.shape[0]  # number of angles
    
    logger.info(f"Saving {num_projections} projections as {file_format.upper()}...")
    logger.info(f"Each projection size: {projections.shape[1]} × {projections.shape[2]} (H×W)")
    
    for i in tqdm(range(num_projections), desc="Saving projections"):
        proj = projections[i, :, :]  # Extract single projection (rows, cols)
        
        if file_format.lower() == "tiff":
            # Save as 32-bit float TIFF
            try:
                import tifffile
                filename = os.path.join(slices_folder, f"projection_{i:04d}.tiff")
                tifffile.imwrite(filename, proj.astype(np.float32))
            except ImportError:
                # Fallback to PIL (converts to uint16)
                proj_norm = ((proj - proj.min()) / (proj.max() - proj.min()) * 65535).astype(np.uint16)
                img = Image.fromarray(proj_norm)
                filename = os.path.join(slices_folder, f"projection_{i:04d}.tiff")
                img.save(filename)
        
        elif file_format.lower() == "png":
            # Normalize to uint16 for PNG
            proj_norm = ((proj - proj.min()) / (proj.max() - proj.min()) * 65535).astype(np.uint16)
            img = Image.fromarray(proj_norm)
            filename = os.path.join(slices_folder, f"projection_{i:04d}.png")
            img.save(filename)
        
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
    logger.info(f"Projections saved to {slices_folder}")


def save_metadata(config, output_folder):
    """
    Save metadata.json for reconstruction pipeline
    """
    metadata_path = os.path.join(output_folder, "slices", "metadata.json")
    
    with open(metadata_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_path}")


def create_dataset(output_folder="CBCT_3D_SheppLogan", config_preset="scaled"):
    """
    Create complete CBCT Shepp-Logan dataset
    
    config_preset: "scaled" (recommended) or "full" (high-res)
    """
    
    if config_preset == "scaled":
        # Scaled configuration - faster, suitable for testing
        config = {
            "name": "CBCT_3D_SheppLogan_Scaled",
            "description": "Synthetic cone-beam CT of 3D Shepp-Logan phantom (scaled, optimized geometry)",
            
            "num_projections": 1440,
            "angle_step": 0.25,
            "start_angle": 0.0,
            "scan_angle_degrees": 360.0,
            "acquisition_direction": "CCW",
            
            "detector_pixels": [768, 768],
            "detector_size_mm": [250.0, 250.0],  # Reduced from 430mm
            "pixel_size_mm": [0.3255, 0.3255],   # 250/768
            "detector_offset": [0.0, 0.0],
            "detector_offset_u": 0.0,
            "detector_offset_v": 0.0,
            
            "source_origin_dist": 800.0,         # Increased from 300mm
            "source_detector_dist": 1000.0,      # Increased from 645mm
            "magnification": 1.25,                # Reduced from 2.15
            
            "volume_voxels": [256, 256, 256],
            "volume_size_mm": [160.0, 160.0, 160.0],  # Fits in 200mm FOV
            "voxel_size_mm": [0.625, 0.625, 0.625],   # 160/256
            
            "projection_dtype": "float32",
            "file_format": "tiff",
            
            "astra_downsample_factor": 1,
            
            "dark_current": 0.0,
            "apply_log_correction": False,
            "apply_bad_pixel_correction": False,
            "apply_noise_reduction": False,
            "apply_truncation_correction": False,
            
            "notes": "Synthetic dataset - no preprocessing needed"
        }
        phantom_size = 256
        
    elif config_preset == "full":
        # Full resolution - matches real data more closely
        config = {
            "name": "CBCT_3D_SheppLogan_Full",
            "description": "Synthetic cone-beam CT of 3D Shepp-Logan phantom (full resolution)",
            
            "num_projections": 720,
            "angle_step": 0.5,
            "start_angle": 0.0,
            "scan_angle_degrees": 360.0,
            "acquisition_direction": "CCW",
            
            "detector_pixels": [1536, 1536],
            "detector_size_mm": [430.0, 430.0],
            "pixel_size_mm": [0.2799, 0.2799],
            "detector_offset": [0.0, 0.0],
            "detector_offset_u": 0.0,
            "detector_offset_v": 0.0,
            
            "source_origin_dist": 800.0,
            "source_detector_dist": 1000.0,
            "magnification": 1.25,
            
            "volume_voxels": [512, 512, 512],
            "volume_size_mm": [180.0, 180.0, 180.0],
            "voxel_size_mm": [0.3515625, 0.3515625, 0.3515625],
            
            "projection_dtype": "float32",
            "file_format": "tiff",
            
            "astra_downsample_factor": 2,
            
            "dark_current": 0.0,
            "apply_log_correction": False,
            "apply_bad_pixel_correction": False,
            "apply_noise_reduction": False,
            "apply_truncation_correction": False,
            
            "notes": "Synthetic dataset - high resolution, use downsample_factor=2"
        }
        phantom_size = 512
    
    else:
        raise ValueError(f"Unknown preset: {config_preset}")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate phantom
    phantom = create_3d_shepplogan_phantom(phantom_size)
    
    # Generate projections
    projections = generate_cbct_projections(phantom, config)
    
    # Save projections
    save_projections(projections, output_folder, config["file_format"])
    
    # Save metadata
    save_metadata(config, output_folder)
    
    logger.info(f"Dataset creation complete: {output_folder}")
    logger.info(f"Ready for reconstruction with your ASTRA pipeline")
    
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CBCT 3D Shepp-Logan dataset")
    parser.add_argument("--output", "-o", default="data/CBCT_3D_SheppLogan",
                        help="Output folder name")
    parser.add_argument("--preset", "-p", choices=["scaled", "full"], default="full",
                        help="Configuration preset: 'scaled' (faster) or 'full' (high-res)")
    
    args = parser.parse_args()
    
    try:
        config = create_dataset(args.output, args.preset)
        
        logger.info("\n" + "="*60)
        logger.info("Dataset generation successful!")
        logger.info(f"Location: {args.output}/slices/")
        logger.info(f"Projections: {config['num_projections']}")
        logger.info(f"Detector: {config['detector_pixels']}")
        logger.info(f"Volume: {config['volume_voxels']}")
        logger.info("\nTo reconstruct:")
        logger.info(f"  python your_astra_script.py --input {args.output}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()