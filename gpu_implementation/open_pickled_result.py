# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# import os

# def interactive_view_pickle(pickle_path: str, axis: int = 0, norm_mode: str = "auto"):
#     """
#     Open a pickled 3D volume and view it slice-by-slice interactively.

#     Args:
#         pickle_path (str): Path to the pickled 3D numpy array.
#         axis (int): Axis along which to slice (default 0 = z-axis).
#         norm_mode (str): 
#             "auto" = per-slice normalization (each slice uses its own min/max)
#             "global" = global normalization (whole volume min/max)
#             "none" = no normalization (may produce bad contrast)
#     """
#     with open(pickle_path, 'rb') as f:
#         volume = pickle.load(f)
#     if not isinstance(volume, np.ndarray):
#         raise ValueError("Loaded object is not a NumPy array. Got type: {}".format(type(volume)))
#     if volume.ndim != 3:
#         raise ValueError("Expected a 3D numpy array. Got shape: {}".format(volume.shape))
#     num_slices = volume.shape[axis]

#     # For global normalization if needed
#     if norm_mode == "global":
#         vmin, vmax = np.min(volume), np.max(volume)
#         if vmax == vmin:
#             vmax = vmin + 1

#     # Initial slice
#     idx = 0

#     def get_slice(idx):
#         if axis == 0:
#             return volume[idx, :, :]
#         elif axis == 1:
#             return volume[:, idx, :]
#         elif axis == 2:
#             return volume[:, :, idx]

#     fig, ax = plt.subplots()
#     plt.title(f"Slice {idx+1}/{num_slices} (axis={axis})")
#     slice_2d = get_slice(idx)
#     if norm_mode == "auto":
#         vmin, vmax = np.min(slice_2d), np.max(slice_2d)
#         if vmax == vmin:
#             vmax = vmin + 1
#     elif norm_mode == "global":
#         pass  # Already set
#     elif norm_mode == "none":
#         vmin, vmax = 0, 255
#     else:
#         raise ValueError("norm_mode must be 'auto', 'global', or 'none'")

#     im = ax.imshow(slice_2d, cmap="gray", vmin=vmin, vmax=vmax)
#     ax.set_xlabel(f"Use Left/Right arrow keys or 'a'/'d' to navigate. ESC or q to quit.")

#     def on_key(event):
#         nonlocal idx
#         if event.key in ['right', 'd']:
#             idx = (idx + 1) % num_slices
#         elif event.key in ['left', 'a']:
#             idx = (idx - 1) % num_slices
#         elif event.key in ['escape', 'q']:
#             plt.close(fig)
#             return
#         else:
#             return

#         slice_2d = get_slice(idx)
#         if norm_mode == "auto":
#             vmin_, vmax_ = np.min(slice_2d), np.max(slice_2d)
#             if vmax_ == vmin_:
#                 vmax_ = vmin_ + 1
#         elif norm_mode == "global":
#             vmin_, vmax_ = vmin, vmax
#         elif norm_mode == "none":
#             vmin_, vmax_ = 0, 255

#         im.set_data(slice_2d)
#         im.set_clim(vmin_, vmax_)
#         ax.set_title(f"Slice {idx+1}/{num_slices} (axis={axis})")
#         fig.canvas.draw_idle()

#     fig.canvas.mpl_connect('key_press_event', on_key)
#     plt.show()

# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser(description="Interactively view slices from a pickled 3D numpy array.")
#     # parser.add_argument("pickle_path", help="Path to the pickled 3D numpy array (e.g., 'volume.pickle').")
#     # parser.add_argument("--axis", type=int, default=0, help="Axis to slice along (0=z, 1=y, 2=x). Default: 0")
#     # parser.add_argument("--norm_mode", type=str, choices=["auto", "global", "none"], default="auto", help="Normalization mode: auto (per-slice), global (whole volume), none (raw).")
#     # args = parser.parse_args()

#     # interactive_view_pickle(args.pickle_path, args.axis, args.norm_mode)
#     interactive_view_pickle(pickle_path="data/20200225_AXI_final_code/results/volume.pickle", axis=2, norm_mode="auto")


import pickle
import numpy as np
import napari

def view_pickled_volume_napari(path: str):
    """
    Load a pickled 3D numpy array and visualize it using napari.
    """
    # Path to the pickled volume
    path = 'data/20200225_AXI_final_code/results/volume.pickle'
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
    view_pickled_volume_napari(path='data/20200225_AXI_final_code/results/volume.pickle')


