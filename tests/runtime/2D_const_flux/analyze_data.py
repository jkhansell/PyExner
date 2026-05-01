import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# --------------------------------------------------
# LOAD
# --------------------------------------------------
ds = xr.open_dataset("analytical_case_out.nc")
print(ds)

outdir = "results_simple"
os.makedirs(outdir, exist_ok=True)

# --------------------------------------------------
# ANALYTICAL
# --------------------------------------------------
def analytical_solution(
    x, t,
    q=1.0,
    Ag=0.005,
    alpha=0.005,
    beta=0.005,
    gamma=1.0,
    p=1.5,
    g=9.81
):
    ue2 = ((alpha*x + beta)/Ag)**(1.0/p)
    u = np.sqrt(ue2)
    h = q/u
    zb0 = -(u**3 + 2*q*g)/(2*u*g) + gamma
    zb = -alpha*t + zb0
    hu = q*np.ones_like(x)
    return h, u, hu, zb


def nearest_time_index(t):
    tt = ds.t.values
    return np.argmin(np.abs(tt - t))


# --------------------------------------------------
# PROFILES
# --------------------------------------------------
def plot_profiles(t=6.0):

    k = nearest_time_index(t)

    x = ds.x.values
    ymid = len(ds.y)//2

    h_num  = ds.h[k, ymid, :].values
    hu_num = ds.hu[k, ymid, :].values
    zb_num = ds.z_b[k, ymid, :].values
    u_num  = hu_num / np.maximum(h_num, 1e-12)

    h_ex, u_ex, hu_ex, zb_ex = analytical_solution(x, t)

    fig, ax = plt.subplots(2,2, figsize=(12,8), dpi=220)

    ax[0,0].plot(x,h_num,lw=2,label="Num")
    ax[0,0].plot(x,h_ex,"--",label="Exact")
    ax[0,0].set_title("Depth h")

    ax[0,1].plot(x,u_num,lw=2,label="Num")
    ax[0,1].plot(x,u_ex,"--",label="Exact")
    ax[0,1].set_title("Velocity u")

    ax[1,0].plot(x,hu_num,lw=2,label="Num")
    ax[1,0].plot(x,hu_ex,"--",label="Exact")
    ax[1,0].set_title("Discharge hu")

    ax[1,1].plot(x,zb_num,lw=2,label="Num")
    ax[1,1].plot(x,zb_ex,"--",label="Exact")
    ax[1,1].set_title("Bed zb")

    for a in ax.flat:
        a.grid()
        a.legend()
        a.set_xlabel("x")

    plt.suptitle(f"Berthon Profiles t={t}")
    plt.tight_layout()

    fname = os.path.join(outdir, f"profiles_t_{t:.2f}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# ERROR VS TIME
# --------------------------------------------------
def plot_error_vs_time():

    x = ds.x.values
    ymid = len(ds.y)//2
    times = ds.t.values

    eh, ez = [], []

    for t in times:

        k = nearest_time_index(float(t))

        h_num  = ds.h[k, ymid, :].values
        zb_num = ds.z_b[k, ymid, :].values

        h_ex, _, _, zb_ex = analytical_solution(x, float(t))

        eh.append(np.mean(np.abs(h_num-h_ex)))
        ez.append(np.mean(np.abs(zb_num-zb_ex)))

    plt.figure(figsize=(10,5), dpi=220)
    plt.plot(times, eh, lw=2, label="mean |h-error|")
    plt.plot(times, ez, lw=2, label="mean |zb-error|")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.xlabel("time")
    plt.title("Error vs Time")

    fname = os.path.join(outdir, "error_vs_time.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# HEATMAP
# --------------------------------------------------
def plot_heatmap(t=7.0, field="h"):

    k = nearest_time_index(t)
    arr = ds[field][k,:,:].values

    x = ds.x.values
    y = ds.y.values

    plt.figure(figsize=(8,3), dpi=220)
    pm = plt.pcolormesh(x, y, arr, shading="auto")
    plt.colorbar(pm, label=field)
    plt.title(f"{field} at t={t}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.gca().set_aspect("equal")

    fname = os.path.join(outdir, f"{field}_t_{t:.2f}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# RUN
# --------------------------------------------------
t = 10.0
plot_profiles(t)
plot_heatmap(t, "h")
plot_heatmap(t, "hu")
plot_heatmap(t, "hv")
plot_heatmap(t, "z_b")

print("Saved figures in:", outdir)