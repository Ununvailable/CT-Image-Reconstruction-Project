import pickle
import numpy as np
import napari

def read_pickled_config(path: str):
    """
    Read a pickled configuration dictionary.
    """
    with open(path, 'rb') as f:
        config = str(pickle.load(f))

    return config

def view_pickled_volume_napari(path: str):
    """
    Load a pickled 3D numpy array and visualize it using napari.
    """
    # Path to the pickled volume
    # path = 'data/20200225_AXI_final_code/results/volume.pickle'
    # Load the volume
    with open(path, 'rb') as f:
        vol = pickle.load(f)

    # Confirm it's a numpy array and 3D
    assert isinstance(vol, np.ndarray) and vol.ndim == 3
    print(f"Loaded volume shape: {vol.shape}, dtype: {vol.dtype}")

    # Launch napari viewer
    viewer = napari.Viewer()
    viewer.add_image(vol, name='Reconstructed CT Volume', colormap='viridis')
    napari.run()

if __name__ == "__main__":
    print(read_pickled_config(path='data/20240530_ITRI_downsampled_4x/results_astra/config_snapshot.pickle'))
    view_pickled_volume_napari(path='data/20240530_ITRI_downsampled_4x/results_astra/volume.pickle')