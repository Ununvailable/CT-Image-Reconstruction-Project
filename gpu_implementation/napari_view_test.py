"""
Launch napari with a reconstructed ASTRA volume and optional PNG slices.

Run:
    conda activate <your-env>
    python view_astra_volume.py
"""

import numpy as np
import napari
from pathlib import Path
from imageio.v3 import imread

# ---------- CONFIGURATION ----------
# Update this to the folder containing your reconstruction output
DATA_DIR = Path(r"D:\CT-Image-Reconstruction-Project\data\astra_reconstructed")
RAW_NAME = "astra_volume.raw"
META_NAME = "astra_metadata.npz"
# -----------------------------------

def load_volume_and_metadata():
    """Load 3-D volume using metadata stored in the npz file."""
    meta = np.load(DATA_DIR / META_NAME)

    # Required metadata fields
    shape = meta["shape"]                       # (Z, Y, X)
    dtype = str(meta["data_type"].item())       # e.g. 'uint16' or 'float32'
    voxel_size = float(meta["voxel_size"].item())  # mm, assume isotropic

    # Load raw volume
    raw_path = DATA_DIR / RAW_NAME
    volume = np.fromfile(raw_path, dtype=dtype).reshape(shape)

    return volume, voxel_size

def load_png_slices():
    """Load optional reference PNG slices if present."""
    slices = {}
    for name in ["slice_axial.png", "slice_coronal.png", "slice_sagittal.png"]:
        path = DATA_DIR / name
        if path.exists():
            slices[name.replace(".png","")] = imread(path)
    return slices

def main():
    volume, voxel_size = load_volume_and_metadata()
    slices = load_png_slices()

    # Start napari viewer with data already added
    viewer = napari.Viewer()

    # Add 3-D reconstruction
    viewer.add_image(
        volume,
        name="Reconstruction Volume",
        scale=(voxel_size, voxel_size, voxel_size),
        rendering="attenuated_mip",
        colormap="gray"
    )

    # Add 2-D reference slices if available
    for name, img in slices.items():
        viewer.add_image(img, name=name, colormap="gray")

    napari.run()

if __name__ == "__main__":
    main()


# Code does not run due to errors in the processing of reconstructed volume. Run the following snippet separately in a napari console to visualize the volume.
''' 
import numpy as np
from pathlib import Path

# Path to your folder
base = Path(r"D:\CT-Image-Reconstruction-Project\data\astra_reconstructed")

# Load metadata
m = np.load(base / "astra_metadata.npz")
shape = m["shape"]                 # e.g. (Z, Y, X)
dtype = str(m["data_type"].item()) # e.g. 'uint16' or 'float32'
voxel_size = m["voxel_size"].item()

# Read the raw volume using correct dtype and shape
vol = np.fromfile(base / "astra_volume.raw", dtype=dtype).reshape(shape)

# Add to the running napari viewer
viewer.add_image(vol, name="Reconstruction Volume",
                 scale=(voxel_size, voxel_size, voxel_size))
'''
