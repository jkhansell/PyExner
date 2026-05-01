import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import os
import imageio

folder = 'results'
os.makedirs(folder, exist_ok=True)

# dataset
ds = Dataset("analytical_case_out.nc")
data = xr.open_dataset(xr.backends.NetCDF4DataStore(ds))

################## Absolute error - analytical vs numerical ##################
# function for the analytical solution
def analytical_solution_2D(Ll=0, Lr=7, dh=0.1, t=0, q=3, A_g=0.005, alpha=0.005, beta=0.005, gamma=1, p=3/2, g=9.81):

    x = np.arange(Ll, Lr + dh, dh)
    y = np.arange(0, 1 + dh, dh)

    X, Y = np.meshgrid(x, y, indexing="xy")

    # velocity squared
    ue2 = ((alpha * X + beta) / A_g)**(1/p)

    # velocity
    u = np.sqrt(ue2)

    # height
    h = q / u

    # sediment z_b0
    zb0 = -(u**3 + 2*q*g) / (2*u*g) + gamma

    # time evolution
    zb = -alpha * t + zb0

    return h, u, zb, zb0, x, y

def get_index_from_time(t, t_array):
    return np.argmin(np.abs(t_array - t))

def fig_2D_comparison(data, t_sel, h_min, h_max, zb_min, zb_max, hu_min, hu_max, err_h_max, err_zb_max, err_hu_max, dh):
    # time
    t_max = 6
    t_num = np.linspace(0, t_max, data.h.shape[0])
    # t_sel = 6 # modify time if needed

    idt = get_index_from_time(t_sel, t_num)

    # data at time slected
    h_exact, u_exact, zb_exact, zb0, x, y = analytical_solution_2D(t=t_sel, dh=dh)

    # [t,y,x] modify
    h_num_0 = data.h[idt,:,:]
    zb0_num_0 = data.z_b[idt,:,:]
    hu_num_0 = data.hu[idt,:,:]


    # figure
    fig, axes = plt.subplots(3, 3, figsize=(22, 9), dpi=200, constrained_layout=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 14
    })
    fig.suptitle(f"Solución analítica vs numérica ($t={t_sel}$)", fontsize=22, fontweight='bold')

    # errors
    hu_exact = h_exact * u_exact

    err_h  = np.abs(h_num_0 - h_exact)
    err_zb = np.abs(zb0_num_0 - zb_exact)
    err_hu = np.abs(hu_num_0 - hu_exact)

    # # scales
    # vmin_h, vmax_h = min(h_exact.min(), h_num_0.min()), max(h_exact.max(), h_num_0.max())
    # vmin_zb, vmax_zb = min(zb_exact.min(), zb0_num_0.min()), max(zb_exact.max(), zb0_num_0.max())
    # vmin_hu, vmax_hu = min(hu_exact.min(), hu_num_0.min()), max(hu_exact.max(), hu_num_0.max())

    # first row: h
    norm_h  = Normalize(vmin=h_min, vmax=h_max)

    im0 = axes[0,0].imshow(h_exact, origin="lower", cmap="jet", norm=norm_h)
    im1 = axes[0,1].imshow(h_num_0, origin="lower", cmap="jet", norm=norm_h)
    fig.colorbar(im1, ax=axes[0,0:2], label="Profundidad del agua $h$ (m)", aspect=10, fraction=0.05, pad=0.04)
    im2 = axes[0,2].imshow(err_h, origin="lower", cmap="inferno", vmin=0)
    fig.colorbar(im2, ax=axes[0,2], label=" |$h_{num}-h_{exact}$| (m)", aspect=10, fraction=0.05, pad=0.04)

    # second row: zb
    norm_zb = Normalize(vmin=zb_min, vmax=zb_max)

    im3 = axes[1,0].imshow(zb_exact, origin="lower", cmap="jet", norm=norm_zb)
    im4 = axes[1,1].imshow(zb0_num_0, origin="lower", cmap="jet", norm=norm_zb)
    fig.colorbar(im4, ax=axes[1,0:2], label="Elevación del lecho $z_b$ (m)", aspect=10, fraction=0.05, pad=0.04)
    im5 = axes[1,2].imshow(err_zb, origin="lower", cmap="inferno", vmin=0)
    fig.colorbar(im5, ax=axes[1,2], label=r" |$z_{b\:num}-z_{b\:exact}$| (m)", aspect=10, fraction=0.05, pad=0.04)

    # third row: hu
    norm_hu = Normalize(vmin=hu_min, vmax=hu_max)

    im6 = axes[2,0].imshow(hu_exact, origin="lower", cmap="jet", norm=norm_hu)
    im7 = axes[2,1].imshow(hu_num_0, origin="lower", cmap="jet", norm=norm_hu)
    fig.colorbar(im7, ax=axes[2,0:2], label=" Caudal en el canal $q$ (m$^2$/s)", aspect=10, fraction=0.05, pad=0.04)
    im8 = axes[2,2].imshow(err_hu, origin="lower", cmap="inferno", vmin=0)
    fig.colorbar(im8, ax=axes[2,2], label=" |$q_{num}-q_{exact}$| (m$^2$/s)", aspect=10, fraction=0.05, pad=0.04)

    # titles
    col_titles = ["Analítica", "Numérica", "Error absoluto"]
    for j in range(3):
        axes[0,j].set_title(col_titles[j], fontsize=18, pad=10, fontweight='bold')

    row_labels = [r"$h$", r"$z_b$", r"$hu$"]
    for i in range(3):
        axes[i,0].annotate(row_labels[i],
                        xy=(-0.2, 0.5),
                        xycoords='axes fraction',
                        ha='center', va='center',
                        fontsize=20, fontweight='bold')

    # axes
    for ax in axes.flat:
        ax.set_xlabel("$x$ (m)", fontsize=15)
        ax.set_ylabel("$y$ (m)", fontsize=15)
        ax.set_aspect("equal")

    #hlines
    offsets = [0.02, 0.01]
    for i in range(1, 3):
        fig.add_artist(plt.Line2D([0.05, 0.95],
                                [1 - i/3 -offsets[i-1], 1 - i/3 -offsets[i-1]],
                                transform=fig.transFigure,
                                color="gray", linewidth=0.8))

    folder_path = os.path.join(folder, "frames_analytical_vs_numerical")
    os.makedirs(folder_path, exist_ok=True) 
    file_path = os.path.join(folder_path, f"analytical_vs_numerical_{t_sel}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()

    ################## Line profiles ##################
    mid_y = data.h.shape[1] // 2

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200, constrained_layout=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 10
    })

    axes[0,0].plot(x, data.h[idt,mid_y,:],label="Numérica", linestyle="solid", c='black', lw='2')
    axes[0,0].plot(x, h_exact[0, :], label="Analítica", c='r', linestyle="dashed")
    axes[0,0].set_xlabel("$x$ (m)")
    axes[0,0].set_ylabel("$h$ (m)")
    axes[0,0].set_title(f"Perfil 1D de $h$ ($t$ = {t_sel}, $y$ = {mid_y})")
    axes[0,0].legend()
    axes[0,0].grid()

    axes[0,1].plot(x, data.z_b[idt,mid_y,:],label="Numérica", linestyle="solid", c='black', lw='2')
    axes[0,1].plot(x, zb_exact[0, :], label="Analítica", c='r', linestyle="dashed")
    axes[0,1].plot(x, zb0[0, :], label="Inicial", linestyle="dotted")
    axes[0,1].set_xlabel("$x$ (m)")
    axes[0,1].set_ylabel("$z_b$ (m)")
    axes[0,1].set_title(f"Perfil 1D de $z_b$ ($t$ = {t_sel}, $y$ = {mid_y})")
    # axes[0,1].set_ylim(-3, 1)
    axes[0,1].legend()
    axes[0,1].grid()

    axes[1,0].plot(x, data.hu[idt,8,:],label="Numérica", linestyle="solid", c='black', lw='2')
    axes[1,0].plot(x, hu_exact[0, :], label="Analítica", c='r', linestyle="dashed")
    axes[1,0].set_xlabel("$x$ (m)")
    axes[1,0].set_ylabel("$q$ (m$^2$/s)")
    axes[1,0].set_title(f"Perfil 1D de $q$ ($t$ = {t_sel}, $y$ = {mid_y})")
    axes[1,0].legend()
    axes[1,0].grid()

    axes[1,1].plot(x, data.hu[idt,8,:]/data.h[idt,8,:],label="Numérica", linestyle="solid", c='black', lw='2')
    axes[1,1].plot(x, u_exact[0, :], label="Analítica", c='r', linestyle="dashed")
    axes[1,1].set_xlabel("$x$ (m)")
    axes[1,1].set_ylabel("$u$ (m/s)")
    axes[1,1].set_title(f"Perfil 1D de $u$ ($t$ = {t_sel}, $y$ = {mid_y})")
    axes[1,1].legend()
    axes[1,1].grid()
    
    folder_path = os.path.join(folder, "frames_profiles")
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"frames_profile_{t_sel}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()

# create gifs
def make_gif(folder_path, output_name, fps=2):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    
    images = []
    for file in files:
        path = os.path.join(folder_path, file)
        images.append(imageio.imread(path))
    
    output_path = os.path.join(folder_path, output_name)
    imageio.mimsave(output_path, images, fps=fps)


################## Norms ##################
def norms():
    # dh values
    dh_values = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    # lists to save norms
    L1_h, Linf_h = [], []
    L1_zb, Linf_zb = [], []

    for dh in dh_values:

        dh_str = str(dh)
        folder = f"dh_{dh_str}"
        file_path = os.path.join(folder, f"analytical_case_{dh_str}_out.nc")

        ds = xr.open_dataset(file_path)

        # values at t = 6 s
        h_num = ds["h"].isel(t=-1).values
        zb_num = ds["z_b"].isel(t=-1).values

        x = ds["x"].values
        y = ds["y"].values

        # analytical solution
        h_exact, _, zb_exact, _, _, _ = analytical_solution_2D(
            t=float(ds["t"].values[-1]),
            dh=dh
        )

        assert h_num.shape == h_exact.shape

        # errors
        err_h = np.abs(h_num - h_exact)
        err_zb = np.abs(zb_num - zb_exact)

        # norms
        dx = dh
        dy = dh
        cell_area = dx * dy

        L1_h.append(np.sum(err_h) * cell_area)
        Linf_h.append(np.max(err_h))

        L1_zb.append(np.sum(err_zb) * cell_area)
        Linf_zb.append(np.max(err_zb))

    # convert to arrays
    dh_values = np.array(dh_values)

    # figure L1
    plt.figure()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
    })
    plt.loglog(dh_values, L1_h, 'o-', label='Norma $L_1$ $h$', linestyle="dotted")
    plt.loglog(dh_values, L1_zb, 'o-', label='Norma $L_1$ $z_b$', linestyle="dotted")
    plt.gca().invert_xaxis()
    plt.xticks(dh_values, [f"{dh:g}" for dh in dh_values])
    plt.xlabel(r'$\Delta x$ (m)')
    plt.ylabel(r'$L_1$ error (m)')
    plt.legend()
    plt.grid(True)
    
    folder = 'results'
    file_path = os.path.join(folder, f"L_1.png")
    plt.savefig(file_path, dpi=300)
    plt.close()

    # figure L_inf
    plt.figure()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
    })
    plt.loglog(dh_values, Linf_h, 'o-', label='Norma $L_\infty$ $h$', linestyle="dotted")
    plt.loglog(dh_values, Linf_zb, 'o-', label='Norma $L_\infty$ $z_b$', linestyle="dotted")
    plt.gca().invert_xaxis()
    plt.xticks(dh_values, [f"{dh:g}" for dh in dh_values])
    plt.xlabel(r'$\Delta x$ (m)')
    plt.ylabel(r'$L_\infty$ error (m)')
    plt.legend()
    plt.grid(True)

    folder = 'results'
    file_path = os.path.join(folder, f"L_inf.png")
    plt.savefig(file_path, dpi=300)
    plt.close()



# # run
times = np.linspace(0, 6, data.t.shape[0])

# fixed scales
h_min, h_max = np.inf, -np.inf
zb_min, zb_max = np.inf, -np.inf
hu_min, hu_max = np.inf, -np.inf

err_h_max = 0
err_zb_max = 0
err_hu_max = 0

for t in times:
    h_exact, u_exact, zb_exact, zb0, x, y = analytical_solution_2D(t=t, dh=0.00625)
    hu_exact = h_exact * u_exact

    idx = get_index_from_time(t, times)

    h_num = data.h[idx,:,:]
    zb_num = data.z_b[idx,:,:]
    hu_num = data.hu[idx,:,:]

    h_min = min(h_min, h_exact.min(), h_num.min())
    h_max = max(h_max, h_exact.max(), h_num.max())

    zb_min = min(zb_min, zb_exact.min(), zb_num.min())
    zb_max = max(zb_max, zb_exact.max(), zb_num.max())

    hu_min = min(hu_min, hu_exact.min(), hu_num.min())
    hu_max = max(hu_max, hu_exact.max(), hu_num.max())

    err_h  = np.abs(h_num - h_exact)
    err_zb = np.abs(zb_num - zb_exact)
    err_hu = np.abs(hu_num - hu_exact)

    err_h_max  = max(err_h_max, err_h.max())
    err_zb_max = max(err_zb_max, err_zb.max())
    err_hu_max = max(err_hu_max, err_hu.max())

err_h_max  = max(err_h_max, 1e-12)
err_zb_max = max(err_zb_max, 1e-12)
err_hu_max = max(err_hu_max, 1e-12)

# plots
for i in times:
    fig_2D_comparison(data, t_sel=i, h_min=h_min, h_max=h_max,
        zb_min=zb_min, zb_max=zb_max,
        hu_min=hu_min, hu_max=hu_max,
        err_h_max=err_h_max,
        err_zb_max=err_zb_max,
        err_hu_max=err_hu_max, dh=0.00625)


# fig_2D_comparison(data, t_sel=0, h_min=h_min, h_max=h_max,
#         zb_min=zb_min, zb_max=zb_max,
#         hu_min=hu_min, hu_max=hu_max,
#         err_h_max=err_h_max,
#         err_zb_max=err_zb_max,
#         err_hu_max=err_hu_max)


make_gif("results/frames_analytical_vs_numerical", "comparison.gif", fps=1)
make_gif("results/frames_profiles", "profiles.gif")


norms()