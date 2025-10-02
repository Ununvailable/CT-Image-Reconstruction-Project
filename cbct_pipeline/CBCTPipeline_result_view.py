import pickle
import numpy as np
import napari

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

    # Launch napari viewer
    viewer = napari.Viewer()
    viewer.add_image(vol, name='Reconstructed CT Volume', colormap='viridis')
    napari.run()

if __name__ == "__main__":
    view_pickled_volume_napari(path='data/20240530_ITRI_downsampled_4x/results/volume.pickle')


