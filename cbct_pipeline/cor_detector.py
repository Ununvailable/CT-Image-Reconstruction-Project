#!/usr/bin/env python3
"""
Center of Rotation (COR) Detection for CT Reconstruction
Diagnoses COR misalignment and suggests correction values
"""

import numpy as np
import os
import logging
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_raw(path: str, resolution: Tuple[int, int], bit_depth: str = "uint16", 
             header_bytes: int = 0, endianness: str = "little") -> np.ndarray:
    """Load RAW binary file"""
    base_dtype = np.dtype(bit_depth)
    
    if endianness == "little":
        dtype = base_dtype.newbyteorder('<')
    elif endianness == "big":
        dtype = base_dtype.newbyteorder('>')
    else:
        dtype = base_dtype
    
    height, width = resolution
    
    with open(path, "rb") as f:
        if header_bytes > 0:
            f.seek(header_bytes)
        data = np.frombuffer(f.read(), dtype=dtype)
    
    data = data.reshape((height, width))
    return data.astype(np.float32)


def find_cor_cross_correlation(proj_0: np.ndarray, proj_180: np.ndarray, 
                                 search_range: float = 50.0) -> Tuple[float, np.ndarray]:
    """
    Find COR using cross-correlation between 0° and 180° projections
    
    Args:
        proj_0: Projection at 0°
        proj_180: Projection at 180° (or closest to it)
        search_range: Maximum shift to search in pixels
        
    Returns:
        cor_offset: Offset in pixels from detector center (positive = shift right)
        correlation_curve: Array of correlation values for visualization
    """
    # Flip 180° projection horizontally (it should be mirror of 0°)
    proj_180_flip = np.fliplr(proj_180)
    
    # Use central rows for better accuracy (avoid edge artifacts)
    rows = proj_0.shape[0]
    central_start = rows // 4
    central_end = 3 * rows // 4
    
    proj_0_central = proj_0[central_start:central_end, :]
    proj_180_central = proj_180_flip[central_start:central_end, :]
    
    # Normalize projections
    proj_0_norm = (proj_0_central - proj_0_central.mean()) / (proj_0_central.std() + 1e-10)
    proj_180_norm = (proj_180_central - proj_180_central.mean()) / (proj_180_central.std() + 1e-10)
    
    # Search for best alignment
    shifts = np.arange(-search_range, search_range, 0.1)
    correlations = []
    
    for shift in shifts:
        # Shift in pixels (can be fractional)
        shift_int = int(shift)
        
        if shift_int > 0:
            p0 = proj_0_norm[:, shift_int:]
            p180 = proj_180_norm[:, :-shift_int] if shift_int > 0 else proj_180_norm
        elif shift_int < 0:
            p0 = proj_0_norm[:, :shift_int]
            p180 = proj_180_norm[:, -shift_int:]
        else:
            p0 = proj_0_norm
            p180 = proj_180_norm
        
        # Calculate correlation
        min_width = min(p0.shape[1], p180.shape[1])
        p0 = p0[:, :min_width]
        p180 = p180[:, :min_width]
        
        correlation = np.sum(p0 * p180) / (min_width * p0.shape[0])
        correlations.append(correlation)
    
    correlations = np.array(correlations)
    best_idx = np.argmax(correlations)
    best_shift = shifts[best_idx]
    
    # The COR offset is half the shift (because each projection is shifted from center)
    cor_offset = -best_shift / 2.0
    
    logger.info(f"Best correlation at shift: {best_shift:.2f} pixels")
    logger.info(f"Estimated COR offset: {cor_offset:.2f} pixels from detector center")
    logger.info(f"Correlation peak value: {correlations[best_idx]:.4f}")
    
    return cor_offset, correlations


def find_cor_image_difference(proj_0: np.ndarray, proj_180: np.ndarray,
                                search_range: float = 50.0) -> Tuple[float, np.ndarray]:
    """
    Find COR by minimizing difference between 0° and flipped 180° projections
    Alternative method to cross-correlation
    """
    proj_180_flip = np.fliplr(proj_180)
    
    # Use central rows
    rows = proj_0.shape[0]
    central_start = rows // 4
    central_end = 3 * rows // 4
    
    proj_0_central = proj_0[central_start:central_end, :]
    proj_180_central = proj_180_flip[central_start:central_end, :]
    
    shifts = np.arange(-search_range, search_range, 0.1)
    differences = []
    
    for shift in shifts:
        shift_int = int(shift)
        
        if shift_int > 0:
            p0 = proj_0_central[:, shift_int:]
            p180 = proj_180_central[:, :-shift_int] if shift_int > 0 else proj_180_central
        elif shift_int < 0:
            p0 = proj_0_central[:, :shift_int]
            p180 = proj_180_central[:, -shift_int:]
        else:
            p0 = proj_0_central
            p180 = proj_180_central
        
        min_width = min(p0.shape[1], p180.shape[1])
        p0 = p0[:, :min_width]
        p180 = p180[:, :min_width]
        
        diff = np.sum(np.abs(p0 - p180)) / (min_width * p0.shape[0])
        differences.append(diff)
    
    differences = np.array(differences)
    best_idx = np.argmin(differences)
    best_shift = shifts[best_idx]
    cor_offset = -best_shift / 2.0
    
    logger.info(f"Minimum difference at shift: {best_shift:.2f} pixels")
    logger.info(f"Estimated COR offset (diff method): {cor_offset:.2f} pixels from detector center")
    
    return cor_offset, differences


def plot_cor_analysis(shifts: np.ndarray, correlations: np.ndarray, 
                       cor_offset: float, output_path: str):
    """Plot correlation curve for COR detection"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(shifts, correlations, 'b-', linewidth=2)
    plt.axvline(x=-cor_offset*2, color='r', linestyle='--', linewidth=2, 
                label=f'Best shift: {-cor_offset*2:.2f} px')
    plt.xlabel('Shift (pixels)', fontsize=12)
    plt.ylabel('Cross-correlation', fontsize=12)
    plt.title('COR Detection: Cross-correlation vs Shift', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.subplot(1, 2, 2)
    # Zoom in around peak
    zoom_range = 20
    center_idx = np.argmax(correlations)
    zoom_start = max(0, center_idx - zoom_range)
    zoom_end = min(len(correlations), center_idx + zoom_range)
    
    plt.plot(shifts[zoom_start:zoom_end], correlations[zoom_start:zoom_end], 'b-', linewidth=2)
    plt.axvline(x=-cor_offset*2, color='r', linestyle='--', linewidth=2,
                label=f'COR offset: {cor_offset:.2f} px')
    plt.xlabel('Shift (pixels)', fontsize=12)
    plt.ylabel('Cross-correlation', fontsize=12)
    plt.title('Zoomed View Around Peak', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"COR analysis plot saved to {output_path}")
    plt.close()


def save_comparison_image(proj_0: np.ndarray, proj_180: np.ndarray, 
                          cor_offset: float, output_path: str):
    """Save side-by-side comparison of 0° and corrected 180° projections"""
    proj_180_flip = np.fliplr(proj_180)
    
    # Apply the detected shift
    # shift_pixels = int(-cor_offset * 2)
    shift_pixels = int(-cor_offset * 6)
    if shift_pixels != 0:
        proj_180_shifted = np.roll(proj_180_flip, shift_pixels, axis=1)
    else:
        proj_180_shifted = proj_180_flip
    
    # Normalize for display
    def normalize(img):
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-10)
        return (img_norm * 65535).astype(np.uint16)
    
    # Create comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(normalize(proj_0), cmap='gray')
    axes[0].set_title('Projection at 0°', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(normalize(proj_180_flip), cmap='gray')
    axes[1].set_title('Projection at 180° (flipped)', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(normalize(proj_180_shifted), cmap='gray')
    axes[2].set_title(f'180° After COR Correction\n(shift: {shift_pixels} px)', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Comparison image saved to {output_path}")
    plt.close()
    
    # Also save difference image
    diff_path = output_path.replace('.png', '_difference.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    diff_before = np.abs(proj_0 - proj_180_flip)
    diff_after = np.abs(proj_0 - proj_180_shifted)
    
    im1 = axes[0].imshow(diff_before, cmap='hot')
    axes[0].set_title('Difference BEFORE COR correction', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(diff_after, cmap='hot')
    axes[1].set_title('Difference AFTER COR correction', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(diff_path, dpi=150, bbox_inches='tight')
    logger.info(f"Difference image saved to {diff_path}")
    plt.close()


def analyze_cor(dataset_path: str, output_dir: str = "cor_analysis",
                resolution: Tuple[int, int] = (3072, 3072),
                bit_depth: str = "uint16", num_projections: int = 720):
    """
    Complete COR analysis pipeline
    
    Args:
        dataset_path: Path to directory containing renamed .raw files (0001.raw, 0002.raw, ...)
        output_dir: Directory to save analysis results
        resolution: Detector resolution (height, width)
        bit_depth: Data type of raw files
        num_projections: Total number of projections
    """
    logger.info("="*60)
    logger.info("Center of Rotation (COR) Detection")
    logger.info("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load projections at 0° and 180°
    proj_0_path = os.path.join(dataset_path, "0001.raw")
    proj_180_idx = num_projections // 2 + 1  # For 720 projs: projection 361
    proj_180_path = os.path.join(dataset_path, f"{proj_180_idx:04d}.raw")
    
    if not os.path.exists(proj_0_path):
        raise FileNotFoundError(f"Projection file not found: {proj_0_path}")
    if not os.path.exists(proj_180_path):
        raise FileNotFoundError(f"Projection file not found: {proj_180_path}")
    
    logger.info(f"Loading: {proj_0_path}")
    proj_0 = load_raw(proj_0_path, resolution, bit_depth)
    
    logger.info(f"Loading: {proj_180_path}")
    proj_180 = load_raw(proj_180_path, resolution, bit_depth)
    
    logger.info(f"Projection shapes: {proj_0.shape}, {proj_180.shape}")
    
    # Method 1: Cross-correlation
    logger.info("\n--- Method 1: Cross-correlation ---")
    cor_offset_corr, correlations = find_cor_cross_correlation(proj_0, proj_180)
    
    # Method 2: Difference minimization
    logger.info("\n--- Method 2: Difference minimization ---")
    cor_offset_diff, differences = find_cor_image_difference(proj_0, proj_180)
    
    # Use cross-correlation result (generally more robust)
    cor_offset = cor_offset_corr
    
    # Generate visualizations
    shifts = np.arange(-50, 50, 0.1)
    plot_cor_analysis(shifts, correlations, cor_offset, 
                      os.path.join(output_dir, "cor_correlation_plot.png"))
    
    save_comparison_image(proj_0, proj_180, cor_offset,
                         os.path.join(output_dir, "cor_comparison.png"))
    
    # Save results to text file
    results_path = os.path.join(output_dir, "cor_results.txt")
    with open(results_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COR Detection Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Detector resolution: {resolution[1]} x {resolution[0]} pixels\n")
        f.write(f"Detector center (ideal): {resolution[1]/2:.1f} pixels\n\n")
        f.write(f"Method 1 (Cross-correlation): {cor_offset_corr:.2f} pixels\n")
        f.write(f"Method 2 (Difference min):     {cor_offset_diff:.2f} pixels\n\n")
        f.write(f"RECOMMENDED COR OFFSET: {cor_offset:.2f} pixels\n\n")
        f.write("="*60 + "\n")
        f.write("How to apply this correction:\n")
        f.write("="*60 + "\n\n")
        f.write("In your ASTRA reconstruction script, update the geometry:\n\n")
        f.write("proj_geom = astra.create_proj_geom(\n")
        f.write("    'cone',\n")
        f.write("    pixel_size_v, pixel_size_u,\n")
        f.write("    det_rows, det_cols,\n")
        f.write("    angles,\n")
        f.write("    source_object_dist,\n")
        f.write("    source_detector_dist - source_object_dist,\n")
        f.write(f"    {cor_offset:.2f},  # <-- COR offset in U direction\n")
        f.write("    detector_offset_v\n")
        f.write(")\n\n")
        f.write("Or update your metadata.json:\n")
        f.write('"detector_offset": [%.2f, 0.0]\n' % cor_offset)
    
    logger.info(f"\nResults saved to {results_path}")
    
    logger.info("\n" + "="*60)
    logger.info(f"RECOMMENDED COR OFFSET: {cor_offset:.2f} pixels")
    logger.info("="*60)
    logger.info(f"\nCheck visualization in: {output_dir}/")
    
    return cor_offset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect Center of Rotation (COR) for CT reconstruction")
    parser.add_argument("--input", "-i", required=True,
                       help="Path to directory containing renamed .raw files")
    parser.add_argument("--output", "-o", default="cor_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--resolution", "-r", default="3072,3072",
                       help="Detector resolution as 'height,width' (default: 3072,3072)")
    parser.add_argument("--num-projections", "-n", type=int, default=720,
                       help="Total number of projections (default: 720)")
    parser.add_argument("--bit-depth", "-b", default="uint16",
                       help="Bit depth of raw files (default: uint16)")
    
    args = parser.parse_args()
    
    # Parse resolution
    res_parts = args.resolution.split(',')
    resolution = (int(res_parts[0]), int(res_parts[1]))
    
    try:
        cor_offset = analyze_cor(
            args.input,
            args.output,
            resolution=resolution,
            bit_depth=args.bit_depth,
            num_projections=args.num_projections
        )
        
        print(f"\n✓ COR detection complete!")
        print(f"✓ Recommended offset: {cor_offset:.2f} pixels")
        print(f"✓ See {args.output}/ for detailed results")
        
    except Exception as e:
        logger.error(f"COR detection failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)