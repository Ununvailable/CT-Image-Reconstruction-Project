# CBCT Reconstruction Pipeline

Cone-Beam Computed Tomography (CBCT) reconstruction toolkit with GPU acceleration, automatic geometry calibration, and support for multiple file formats.

## Features

- **Multiple Reconstruction Backends**
  - ASTRA Toolbox (GPU-accelerated FDK, SIRT, CGLS)
  - Custom CPU implementation with vectorized backprojection
  
- **Automatic Geometry Calibration**
  - Center of rotation (COR) estimation
  - Detector tilt and skew correction
  - Source-object distance (SOD) refinement
  - Angular offset calibration

- **Flexible Data Pipeline**
  - Support for RAW, TIFF, PNG, JPEG formats
  - Configurable preprocessing (log correction, noise reduction, bad pixel removal)
  - JSON-based metadata configuration
  - Pickle-based intermediate data caching

- **3D Visualization**
  - Interactive volume viewer using napari

- **Synthetic Phantom Generation**
  - 3D Shepp-Logan phantom for testing and validation

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for GPU acceleration)
- CUDA Toolkit 11.x or 12.x (for GPU features)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cbct-pipeline.git
cd cbct-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Acceleration (Optional)

For GPU-accelerated reconstruction:

```bash
# Install CuPy (match your CUDA version)
# For CUDA 11.x:
pip install cupy-cuda11x

# For CUDA 12.x:
pip install cupy-cuda12x

# Install ASTRA Toolbox with CUDA
conda install -c astra-toolbox astra-toolbox
# OR build from source: https://github.com/astra-toolbox/astra-toolbox
```

## Quick Start

### 1. Prepare Your Data

Organize your dataset:
```
your_dataset/
├── slices/
│   ├── metadata.json      # Geometry and acquisition parameters
│   ├── 0001.raw           # Projection files
│   ├── 0002.raw
│   └── ...
└── results/               # Created automatically
```

### 2. Generate Test Data (Optional)

Create synthetic Shepp-Logan phantom:

```bash
python cbct_pipeline/CBCT_ASTRA_3dSheppLogan_gen.py \
    --output data/test_phantom \
    --preset scaled
```

### 3. Calibrate Geometry

Automatically estimate geometry parameters:

```bash
python cbct_pipeline/CBCTPipeline_geometry_calib.py \
    --dataset data/your_dataset \
    --method coarse_to_fine
```

This generates calibration parameters to add to your `metadata.json`.

### 4. Reconstruct Volume

**Using ASTRA Toolbox (recommended):**

```bash
python cbct_pipeline/CBCT_ASTRA_benchmark.py \
    --input data/your_dataset
```

**Using CPU implementation:**

```bash
python cbct_pipeline/CBCTPipeline_cpu_ver.py \
    --input data/your_dataset
```

### 5. Visualize Results

```bash
python cbct_pipeline/CBCTPipeline_result_view.py
# Opens napari viewer with reconstructed volume
```

## Configuration

### metadata.json Format

Minimal required fields:

```json
{
  "acquisition_num_projections": 720,
  "acquisition_angle_step_deg": 0.5,
  "acquisition_start_angle_deg": 0.0,
  
  "detector_rows_px": 1536,
  "detector_cols_px": 1536,
  "detector_pixel_pitch_u_mm": 0.139,
  "detector_pixel_pitch_v_mm": 0.139,
  
  "source_to_origin_dist_mm": 81.454,
  "source_to_detector_dist_mm": 814.554,
  
  "volume_nx_vox": 512,
  "volume_ny_vox": 512,
  "volume_nz_vox": 512,
  "volume_voxel_pitch_mm": 0.352,
  
  "file_format": "raw",
  "raw_dtype": "uint16",
  "raw_endian": "little"
}
```

Full configuration schema documented in source files.

### Calibration Parameters

After running geometry calibration, add these to `metadata.json`:

```json
{
  "calibration_offset_u_px": 12.345,
  "calibration_offset_v_px": -3.456,
  "calibration_tilt_deg": 0.123,
  "calibration_skew_deg": 0.045,
  "calibration_sod_correction_mm": 0.5
}
```

## Project Structure

```
cbct_pipeline/
├── CBCT_ASTRA_benchmark.py              # Main ASTRA reconstruction
├── CBCTPipeline_cpu_ver.py              # CPU-based reconstruction
├── CBCTPipeline_geometry_calib.py       # Automatic calibration
├── CBCTPipeline_result_view.py          # Volume visualization
├── CBCT_ASTRA_3dSheppLogan_gen.py       # Phantom generation
└── README.md

data/
├── your_dataset/
│   ├── slices/
│   │   └── metadata.json
│   ├── results_astra/                   # ASTRA output
│   └── geometry_calib/                  # Calibration output
└── ...
```

## Algorithms

### Reconstruction Algorithms (ASTRA)

- **FDK_CUDA**: Fast filtered backprojection (recommended)
- **SIRT3D_CUDA**: Simultaneous iterative reconstruction
- **CGLS3D_CUDA**: Conjugate gradient least squares

### Calibration Methods

- **Sequential**: Fast, estimates each parameter independently
- **Coarse-to-fine**: Multi-scale approach (recommended)
- **Joint**: Simultaneous optimization of all parameters

## Performance Tips

1. **Use GPU acceleration**: 10-100× speedup with CUDA
2. **Downsample for testing**: Set `preprocessing_downsample_factor: 2` in metadata
3. **Optimize calibration**: Use `--downsample 2` for faster calibration
4. **Memory management**: Process subsets if RAM limited

## Troubleshooting

### CUDA Out of Memory
- Reduce volume size in metadata.json
- Increase downsampling factor
- Process slices in batches

### ASTRA Import Error
```bash
# Linux: Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH

# Or install via conda
conda install -c astra-toolbox astra-toolbox
```

### Poor Reconstruction Quality
1. Run geometry calibration first
2. Check projection alignment visually
3. Verify metadata parameters match scanner specs
4. Increase number of projections if possible

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{cbct_pipeline,
  title={CBCT Reconstruction Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/cbct-pipeline}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with tests

## Contact

- Email: leminhduc24994@gmail.com

## Acknowledgments

- [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox)
- [CuPy](https://cupy.dev/)
- [napari](https://napari.org/)
