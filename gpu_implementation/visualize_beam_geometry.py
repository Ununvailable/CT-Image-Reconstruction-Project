"""
visualize_beam_geometry.py

Visualize source-detector trajectory and projection mapping for sample object points.

Run with: python visualize_beam_geometry.py
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Metadata (from your file) ---
NumRows = 2860
NumChannelsPerRow = 2860
pixel_u = 0.15000000596046448
pixel_v = 0.15000000596046448
SOD = 28.625365287711134               # Source-Object distance
SDD = 699.9996522369905                # Source-Detector distance
detector_offset_u = 1430.1098329145173
detector_offset_v = 1429.4998776624227
NumProjections = 1600
start_angle = 270.0
scan_angle = 360.0
proj_w = NumChannelsPerRow
proj_h = NumRows

# Derived
origin_detector = SDD - SOD
angles_deg = np.linspace(start_angle, start_angle + scan_angle, NumProjections, endpoint=False)
angles = np.deg2rad(angles_deg)

# Helper: source and detector center positions in XY plane
def source_position(theta):
    # theta in radians
    return np.array([SOD * np.sin(theta), -SOD * np.cos(theta)])  # (x,y)

def detector_center(theta):
    # detector center is further along the same ray from source
    # unit vector from source to origin (object center) is [-sin, cos] (approx)
    sx, sy = source_position(theta)
    # vector from source to object center:
    v_to_obj = -np.array([sx, sy])
    dist_to_obj = np.linalg.norm(v_to_obj)
    # normalized direction
    dir_unit = v_to_obj / (dist_to_obj + 1e-12)
    # detector center position: source + dir_unit * (origin_detector + SOD)
    # equivalently source + dir_unit * SDD
    det = np.array([sx, sy]) + dir_unit * SDD
    return det

# Map a world voxel (x,y,z) to detector (u,v) using your kernel-like formulas
def project_point_to_detector(pt, theta):
    # pt: (x,y,z) in mm (world coordinates)
    xw, yw, zw = pt
    sin_a = np.sin(theta)
    cos_a = np.cos(theta)
    # source coords (as kernel)
    sx = SOD * sin_a
    sy = -SOD * cos_a
    # ray from source to point
    ray_x = xw - sx
    ray_y = yw - sy
    # scalar t following kernel logic
    denom = (SOD + ray_y * cos_a + ray_x * sin_a)
    if np.abs(denom) < 1e-12:
        t = np.inf
    else:
        t = (SOD + origin_detector) / denom

    # detector coordinates in pixels (u across width, v along height)
    det_u = (ray_x * cos_a - ray_y * sin_a) * t / pixel_u + proj_w / 2.0 + detector_offset_u
    det_v = zw * t / pixel_v + proj_h / 2.0 + detector_offset_v

    return det_u, det_v, (sx, sy)

# Sample object points (in mm). Choose the origin and some off-center points.
# NOTE: the object coordinate scale must match pixel_size and SOD units (mm)
pts_mm = [
    (0.0, 0.0, 0.0),           # object center
    (5.0, 0.0, 0.0),           # +x offset 5 mm
    (0.0, 5.0, 0.0),           # +y offset 5 mm
    (2.0, 2.0, 10.0)           # elevated z
]

# Compute source and detector centers
sources = np.array([source_position(t) for t in angles])
dets = np.array([detector_center(t) for t in angles])

# Plot top-down geometry: sources, detector centers, object point(s)
plt.figure(figsize=(8,8))
plt.plot(sources[:,0], sources[:,1], '.', markersize=2, label='Source trajectory')
plt.plot(dets[:,0], dets[:,1], '.', markersize=2, label='Detector centers')
plt.scatter(0,0, c='k', s=50, label='Object center (origin)')
for p in pts_mm:
    plt.scatter(p[0], p[1], s=40, label=f'Pt {p[0]:.1f},{p[1]:.1f}')
plt.gca().set_aspect('equal', 'box')
plt.title('Top-down view: source and detector centers (mm)')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.legend()
plt.grid(True)
plt.show()

# For each sample point, compute projected u across angles and plot
plt.figure(figsize=(10,6))
for idx, p in enumerate(pts_mm):
    us = []
    vs = []
    valid = []
    for t in angles:
        u,v,src = project_point_to_detector(p, t)
        # mark as valid if finite and inside detector pixel range
        us.append(u)
        vs.append(v)
        valid.append(np.isfinite(u) and np.isfinite(v))
    us = np.array(us)
    vs = np.array(vs)
    plt.plot(angles_deg, us, label=f'Pt{idx} u (px)')
plt.title('Detector u (pixel) vs gantry angle (deg)')
plt.xlabel('Angle (deg)')
plt.ylabel('u (pixel coordinate)')
plt.legend()
plt.grid(True)
plt.show()

# Show a numeric example: first, middle, last projection for each point
indices = [0, len(angles)//2, len(angles)-1]
print("Sample numeric detector coordinates (u,v) and source positions (sx,sy) for selected angles:")
for i in indices:
    a_deg = angles_deg[i]
    print(f"\nAngle {a_deg:.1f} deg (index {i}):")
    for j,p in enumerate(pts_mm):
        u,v,src = project_point_to_detector(p, angles[i])
        print(f"  Pt{j} {p} -> u={u:.1f}, v={v:.1f}  | source (x,y)=({src[0]:.3f},{src[1]:.3f})")
