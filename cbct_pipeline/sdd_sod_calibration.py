#!/usr/bin/env python3
"""
CT Geometry Calibration - Verify and correct SDD/SOD parameters
Diagnoses incorrect source-detector and source-object distances
"""

import numpy as np
import os
import logging
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import json

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


def analyze_projection_magnification(proj_0: np.ndarray, proj_90: np.ndarray, 
                                     proj_180: np.ndarray, proj_270: np.ndarray,
                                     pixel_size_mm: float) -> dict:
    """
    Analyze projection magnification to verify geometry
    
    If a feature has the same size in all projections, geometry is likely correct.
    If feature size varies, SDD/SOD is wrong.
    """
    results = {}
    
    # Find the centroid of the sample in each projection
    def find_sample_width(proj):
        """Estimate sample width by analyzing projection profile"""
        # Sum along vertical axis to get horizontal profile
        profile = np.sum(proj, axis=0)
        
        # Find edges (where intensity changes significantly)
        profile_smooth = np.convolve(profile, np.ones(10)/10, mode='same')
        grad = np.abs(np.gradient(profile_smooth))
        
        # Threshold to find edges
        threshold = np.mean(grad) + 2 * np.std(grad)
        edge_pixels = np.where(grad > threshold)[0]
        
        if len(edge_pixels) > 1:
            left_edge = edge_pixels[0]
            right_edge = edge_pixels[-1]
            width_pixels = right_edge - left_edge
            width_mm = width_pixels * pixel_size_mm
            center_pixel = (left_edge + right_edge) / 2
            return width_pixels, width_mm, center_pixel
        else:
            return None, None, None
    
    projs = {
        "0°": proj_0,
        "90°": proj_90,
        "180°": proj_180,
        "270°": proj_270
    }
    
    logger.info("\nProjection magnification analysis:")
    logger.info("="*60)
    
    widths_px = []
    widths_mm = []
    
    for angle, proj in projs.items():
        width_px, width_mm, center = find_sample_width(proj)
        if width_px is not None:
            logger.info(f"{angle:5s}: Sample width = {width_px:.1f} px ({width_mm:.2f} mm), Center = {center:.1f} px")
            widths_px.append(width_px)
            widths_mm.append(width_mm)
        else:
            logger.warning(f"{angle:5s}: Could not detect sample edges")
    
    if len(widths_px) > 0:
        mean_width_px = np.mean(widths_px)
        std_width_px = np.std(widths_px)
        variation_pct = (std_width_px / mean_width_px) * 100
        
        logger.info("="*60)
        logger.info(f"Mean width: {mean_width_px:.1f} ± {std_width_px:.1f} pixels")
        logger.info(f"Variation: {variation_pct:.2f}%")
        
        if variation_pct > 5:
            logger.warning("⚠ Large variation detected! SDD/SOD likely incorrect!")
            logger.warning("  Sample size should be consistent across all angles.")
        else:
            logger.info("✓ Low variation - geometry appears correct")
        
        results = {
            "mean_width_pixels": mean_width_px,
            "std_width_pixels": std_width_px,
            "variation_percent": variation_pct,
            "widths": dict(zip(projs.keys(), widths_px))
        }
    
    return results


def check_detector_coverage(proj: np.ndarray, threshold_percentile: float = 95) -> dict:
    """
    Check if sample extends beyond detector FOV
    This is critical - truncation causes severe artifacts
    """
    logger.info("\nDetector coverage analysis:")
    logger.info("="*60)
    
    # Check edge intensities
    top_edge = np.mean(proj[0:50, :])
    bottom_edge = np.mean(proj[-50:, :])
    left_edge = np.mean(proj[:, 0:50])
    right_edge = np.mean(proj[:, -50:])
    center = np.mean(proj[proj.shape[0]//4:-proj.shape[0]//4, 
                          proj.shape[1]//4:-proj.shape[1]//4])
    
    # Calculate threshold
    threshold = np.percentile(proj, threshold_percentile)
    
    results = {
        "top_edge": top_edge,
        "bottom_edge": bottom_edge,
        "left_edge": left_edge,
        "right_edge": right_edge,
        "center": center,
        "threshold": threshold
    }
    
    logger.info(f"Edge intensities (higher = more sample):")
    logger.info(f"  Top:    {top_edge:.1f}")
    logger.info(f"  Bottom: {bottom_edge:.1f}")
    logger.info(f"  Left:   {left_edge:.1f}")
    logger.info(f"  Right:  {right_edge:.1f}")
    logger.info(f"  Center: {center:.1f}")
    
    truncated = False
    if any([top_edge > threshold*0.3, bottom_edge > threshold*0.3, 
            left_edge > threshold*0.3, right_edge > threshold*0.3]):
        logger.warning("⚠ TRUNCATION DETECTED!")
        logger.warning("  Sample extends beyond detector FOV.")
        logger.warning("  This causes severe ring artifacts and reconstruction errors!")
        truncated = True
    else:
        logger.info("✓ Sample appears to fit within detector FOV")
    
    results["truncated"] = truncated
    
    return results


def estimate_correct_sdd_sod(detector_size_mm: float, detector_pixels: int,
                             sample_size_proj_pixels: float, 
                             sample_size_actual_mm: float,
                             current_sod: float) -> Tuple[float, float]:
    """
    Estimate correct SDD/SOD based on measured magnification
    
    Args:
        detector_size_mm: Physical detector width in mm
        detector_pixels: Detector width in pixels
        sample_size_proj_pixels: Sample width in projection (pixels)
        sample_size_actual_mm: Known actual sample size (mm)
        current_sod: Current SOD estimate
        
    Returns:
        Estimated (SOD, SDD)
    """
    pixel_size_mm = detector_size_mm / detector_pixels
    sample_size_proj_mm = sample_size_proj_pixels * pixel_size_mm
    
    # Magnification = projected size / actual size
    measured_magnification = sample_size_proj_mm / sample_size_actual_mm
    
    logger.info(f"\nMagnification calculation:")
    logger.info(f"  Sample in projection: {sample_size_proj_mm:.2f} mm ({sample_size_proj_pixels:.1f} pixels)")
    logger.info(f"  Sample actual size: {sample_size_actual_mm:.2f} mm")
    logger.info(f"  Measured magnification: {measured_magnification:.3f}x")
    
    # For cone beam: M = SDD / SOD
    # So: SOD = SDD / M, or SDD = SOD * M
    
    # If we trust the current SOD estimate:
    estimated_sdd_from_sod = current_sod * measured_magnification
    
    logger.info(f"\nIf SOD = {current_sod:.2f} mm is correct:")
    logger.info(f"  Then SDD should be: {estimated_sdd_from_sod:.2f} mm")
    
    return current_sod, estimated_sdd_from_sod


def visualize_geometry_check(proj_0: np.ndarray, proj_180: np.ndarray,
                             output_path: str):
    """Create visualization for geometry verification"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Normalize for display
    def normalize(img):
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-10)
        return img_norm
    
    # Show projections
    axes[0, 0].imshow(normalize(proj_0), cmap='gray')
    axes[0, 0].set_title('Projection at 0°', fontsize=14)
    axes[0, 0].axhline(y=proj_0.shape[0]//2, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(x=proj_0.shape[1]//2, color='r', linestyle='--', alpha=0.5)
    
    axes[0, 1].imshow(normalize(proj_180), cmap='gray')
    axes[0, 1].set_title('Projection at 180°', fontsize=14)
    axes[0, 1].axhline(y=proj_180.shape[0]//2, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(x=proj_180.shape[1]//2, color='r', linestyle='--', alpha=0.5)
    
    # Horizontal profiles
    profile_0 = np.sum(proj_0, axis=0)
    profile_180 = np.sum(proj_180, axis=0)
    
    axes[1, 0].plot(profile_0, label='0°')
    axes[1, 0].plot(profile_180, label='180°', alpha=0.7)
    axes[1, 0].set_title('Horizontal Profiles (should match if geometry correct)', fontsize=12)
    axes[1, 0].set_xlabel('Pixel position')
    axes[1, 0].set_ylabel('Intensity sum')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Vertical profiles
    profile_0_v = np.sum(proj_0, axis=1)
    profile_180_v = np.sum(proj_180, axis=1)
    
    axes[1, 1].plot(profile_0_v, label='0°')
    axes[1, 1].plot(profile_180_v, label='180°', alpha=0.7)
    axes[1, 1].set_title('Vertical Profiles', fontsize=12)
    axes[1, 1].set_xlabel('Pixel position')
    axes[1, 1].set_ylabel('Intensity sum')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Geometry visualization saved to {output_path}")
    plt.close()


def analyze_geometry(dataset_path: str, metadata_path: str, output_dir: str = "geometry_analysis",
                     resolution: Tuple[int, int] = (3072, 3072),
                     bit_depth: str = "uint16", num_projections: int = 720,
                     known_sample_size_mm: Optional[float] = None):
    """
    Complete geometry analysis pipeline
    
    Args:
        dataset_path: Path to directory containing .raw files
        metadata_path: Path to metadata.json
        output_dir: Output directory
        resolution: Detector resolution
        bit_depth: Data type
        num_projections: Total projections
        known_sample_size_mm: If you know actual sample size, provide it here
    """
    logger.info("="*60)
    logger.info("CT Geometry Verification")
    logger.info("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    current_sod = metadata.get("source_origin_dist", 0)
    current_sdd = metadata.get("source_detector_dist", 0)
    detector_size_mm = metadata.get("detector_size_mm", [430, 430])[0]
    pixel_size_mm = metadata.get("pixel_size_mm", [0.139, 0.139])[0]
    
    logger.info(f"\nCurrent metadata values:")
    logger.info(f"  SOD (source-object): {current_sod:.3f} mm")
    logger.info(f"  SDD (source-detector): {current_sdd:.3f} mm")
    logger.info(f"  Calculated magnification: {current_sdd/current_sod:.3f}x")
    logger.info(f"  Detector size: {detector_size_mm:.1f} mm")
    logger.info(f"  Pixel size: {pixel_size_mm:.3f} mm")
    
    # Load key projections
    proj_files = {
        "0°": "0001.raw",
        "90°": f"{num_projections//4 + 1:04d}.raw",
        "180°": f"{num_projections//2 + 1:04d}.raw",
        "270°": f"{3*num_projections//4 + 1:04d}.raw"
    }
    
    projections = {}
    for angle, filename in proj_files.items():
        filepath = os.path.join(dataset_path, filename)
        if os.path.exists(filepath):
            logger.info(f"Loading {angle} projection: {filename}")
            projections[angle] = load_raw(filepath, resolution, bit_depth)
        else:
            logger.warning(f"File not found: {filepath}")
    
    if len(projections) < 2:
        raise RuntimeError("Need at least 2 projections for analysis")
    
    # Check detector coverage
    logger.info("\n" + "="*60)
    coverage = check_detector_coverage(projections["0°"])
    
    # Analyze magnification consistency
    logger.info("\n" + "="*60)
    if len(projections) >= 4:
        mag_results = analyze_projection_magnification(
            projections["0°"], projections["90°"],
            projections["180°"], projections["270°"],
            pixel_size_mm
        )
        
        # If user provided known sample size, estimate geometry
        if known_sample_size_mm and mag_results:
            logger.info("\n" + "="*60)
            mean_width_px = mag_results["mean_width_pixels"]
            estimated_sod, estimated_sdd = estimate_correct_sdd_sod(
                detector_size_mm, resolution[1],
                mean_width_px, known_sample_size_mm,
                current_sod
            )
            
            logger.info("\n" + "="*60)
            logger.info("RECOMMENDED CORRECTIONS:")
            logger.info("="*60)
            logger.info(f"Update metadata.json with:")
            logger.info(f'  "source_origin_dist": {estimated_sod:.3f},')
            logger.info(f'  "source_detector_dist": {estimated_sdd:.3f},')
    
    # Create visualizations
    visualize_geometry_check(projections["0°"], projections["180°"],
                            os.path.join(output_dir, "geometry_check.png"))
    
    # Save report
    report_path = os.path.join(output_dir, "geometry_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CT Geometry Verification Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Metadata: {metadata_path}\n\n")
        f.write(f"Current geometry:\n")
        f.write(f"  SOD: {current_sod:.3f} mm\n")
        f.write(f"  SDD: {current_sdd:.3f} mm\n")
        f.write(f"  Magnification: {current_sdd/current_sod:.3f}x\n\n")
        f.write(f"Truncation detected: {coverage.get('truncated', 'Unknown')}\n\n")
        
        if 'variation_percent' in mag_results:
            f.write(f"Sample width variation: {mag_results['variation_percent']:.2f}%\n")
            f.write(f"  (>5% suggests incorrect SDD/SOD)\n\n")
        
        f.write("="*60 + "\n")
        f.write("NEXT STEPS:\n")
        f.write("="*60 + "\n")
        f.write("1. Check if sample is truncated (extends beyond detector)\n")
        f.write("   - If YES: Reduce magnification or accept artifacts\n")
        f.write("2. If width variation > 5%: SDD/SOD values are likely wrong\n")
        f.write("   - Contact system manufacturer for correct values\n")
        f.write("   - OR measure with calibration phantom\n")
        f.write("3. If you know the actual sample size, run with --sample-size flag\n")
    
    logger.info(f"\nReport saved to {report_path}")
    logger.info("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify CT geometry parameters (SDD/SOD)")
    parser.add_argument("--input", "-i", required=True,
                       help="Path to directory containing .raw files")
    parser.add_argument("--metadata", "-m", required=True,
                       help="Path to metadata.json file")
    parser.add_argument("--output", "-o", default="geometry_analysis",
                       help="Output directory")
    parser.add_argument("--resolution", "-r", default="3072,3072",
                       help="Detector resolution (default: 3072,3072)")
    parser.add_argument("--num-projections", "-n", type=int, default=720,
                       help="Total projections (default: 720)")
    parser.add_argument("--sample-size", "-s", type=float, default=None,
                       help="Known sample size in mm (for geometry calibration)")
    
    args = parser.parse_args()
    
    res_parts = args.resolution.split(',')
    resolution = (int(res_parts[0]), int(res_parts[1]))
    
    try:
        analyze_geometry(
            args.input,
            args.metadata,
            args.output,
            resolution=resolution,
            num_projections=args.num_projections,
            known_sample_size_mm=args.sample_size
        )
        
        print("\n✓ Geometry analysis complete!")
        print(f"✓ See {args.output}/ for results")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)