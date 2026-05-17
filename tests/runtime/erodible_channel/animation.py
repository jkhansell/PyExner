import os
import numpy as np
import xarray as xr
import pyvista as pv

# =============================================================================
# CONFIG
# =============================================================================

DATASET = "erodible_channel_out.nc"

RESULTS_DIR = "results"
VIDEO_PATH = os.path.join(RESULTS_DIR, "pyvista_dambreak.mp4")

os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Visualization controls
# -----------------------------------------------------------------------------

FRAME_SKIP = 1
WARP_SCALE = 1.0
WATER_ALPHA = 0.65

# Downsampling for performance
DS = 2

# =============================================================================
# LOAD DATA
# =============================================================================

data = xr.open_dataset(DATASET, engine="netcdf4")

x_sl = slice(-2.1, 11.9)
y_sl = slice(1.8, -1.8)

subset = data.sel(x=x_sl, y=y_sl)

# =============================================================================
# COORDINATES
# =============================================================================

X = subset.x.values[::DS]
Y = subset.y.values[::DS]

XX, YY = np.meshgrid(X, Y)

# =============================================================================
# INITIAL STATE
# =============================================================================

h0 = subset.h[0].values[::DS, ::DS]

zbed0 = (
    subset.z_b[0].values[::DS, ::DS]
    + subset.z[0].values[::DS, ::DS]
)

water0 = zbed0 + h0

# =============================================================================
# STRUCTURED GRID
# =============================================================================

bed_grid = pv.StructuredGrid(
    XX,
    YY,
    zbed0 * WARP_SCALE
)

water_grid = pv.StructuredGrid(
    XX,
    YY,
    water0 * WARP_SCALE
)

# Scalars for coloring
bed_grid["bed"] = zbed0.ravel(order="F")
water_grid["water"] = h0.ravel(order="F")

# =============================================================================
# PLOTTER
# =============================================================================

pv.set_plot_theme("document")

plotter = pv.Plotter(
    off_screen=True,
    window_size=(1920, 1080)
)

plotter.open_movie(
    VIDEO_PATH,
    framerate=15,
    quality=9
)

# -----------------------------------------------------------------------------
# Lighting
# -----------------------------------------------------------------------------

plotter.enable_lightkit()

# -----------------------------------------------------------------------------
# Bed mesh
# -----------------------------------------------------------------------------

bed_actor = plotter.add_mesh(
    bed_grid,
    scalars="bed",
    cmap="terrain",
    smooth_shading=True,
    specular=0.3,
    ambient=0.2,
    show_scalar_bar=True,
    scalar_bar_args={
        "title": "Bed elevation [m]"
    }
)

# -----------------------------------------------------------------------------
# Water mesh
# -----------------------------------------------------------------------------

water_actor = plotter.add_mesh(
    water_grid,
    scalars="water",
    cmap="Blues",
    opacity=WATER_ALPHA,
    smooth_shading=True,
    specular=0.8,
    ambient=0.15,
    show_scalar_bar=True,
    scalar_bar_args={
        "title": "Water depth [m]"
    }
)

# =============================================================================
# CAMERA
# =============================================================================

center = (5.0,0.0, 0.0)   # focal point (channel center)
r = 25.0                   # camera distance

az = np.deg2rad(45.0)      # azimuth in XY plane
el = np.deg2rad(30.0)      # elevation above plane

cam_x = center[0] + r * np.cos(el) * np.cos(az)
cam_y = center[1] + r * np.cos(el) * np.sin(az)
cam_z = center[2] + r * np.sin(el)

plotter.camera_position = [
    (cam_x, cam_y, cam_z),  # camera
    center,                 # focal point
    (0.0, 0.0, 1.0)         # Z is up
]

plotter.camera.zoom(1.6)


# =============================================================================
# FIRST FRAME
# =============================================================================

plotter.add_axes(
    line_width=5,
    labels_off=False
)

plotter.write_frame()

# =============================================================================
# ANIMATION LOOP
# =============================================================================

for i in range(0, subset.sizes["t"], FRAME_SKIP):

    print(f"Rendering frame {i}")

    # -------------------------------------------------------------------------
    # Fields
    # -------------------------------------------------------------------------

    h = subset.h[i].values[::DS, ::DS]

    zbed = (
        subset.z_b[i].values[::DS, ::DS]
        + subset.z[i].values[::DS, ::DS]
    )

    water = zbed + h

    # Remove very shallow water visually
    water_masked = water.copy()
    water_masked[h < 1e-3] = np.nan

    # -------------------------------------------------------------------------
    # Update geometry
    # -------------------------------------------------------------------------

    bed_points = np.column_stack([
        XX.ravel(order="F"),
        YY.ravel(order="F"),
        (zbed * WARP_SCALE).ravel(order="F")
    ])

    water_points = np.column_stack([
        XX.ravel(order="F"),
        YY.ravel(order="F"),
        (water_masked * WARP_SCALE).ravel(order="F")
    ])

    bed_grid.points = bed_points
    water_grid.points = water_points

    plotter.add_axes(
        line_width=5,
        labels_off=False
    )

    # -------------------------------------------------------------------------
    # Update scalars
    # -------------------------------------------------------------------------

    bed_grid["bed"] = zbed.ravel(order="F")
    water_grid["water"] = h.ravel(order="F")

    # -------------------------------------------------------------------------
    # Update text
    # -------------------------------------------------------------------------

    current_time = float(subset.t[i].values)

    plotter.add_text(
        f"Time: {current_time:.2f} s",
        position="upper_left",
        font_size=12,
        name="time_label"
    )

    # -------------------------------------------------------------------------
    # Render frame
    # -------------------------------------------------------------------------

    plotter.write_frame()

# =============================================================================
# FINALIZE
# =============================================================================

plotter.close()

print(f"\nAnimation saved to:\n{VIDEO_PATH}")