import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# --------------------------------------------------
# ANALYTICAL SOLUTION
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

def nearest_time_index(t, times):
    return np.argmin(np.abs(times - t))

# --------------------------------------------------
# CONVERGENCE ANALYSIS
# --------------------------------------------------
def calculate_norms(dh_values, target_time=10.0):
    
    L1_h, L2_h, Linf_h = [], [], []
    L1_zb, L2_zb, Linf_zb = [], [], []
    L1_hu, L2_hu, Linf_hu = [], [], []
    valid_dh = []

    for dh in dh_values:
        dh_str = str(dh)
        folder = f"dh_{dh_str}"
        nc_file = os.path.join(folder, f"analytical_case_{dh_str}_out.nc")
        
        if not os.path.exists(nc_file):
            print(f"Warning: Output file {nc_file} not found. Skipping.")
            continue
            
        ds = xr.open_dataset(nc_file)
        
        times = ds.t.values
        k = nearest_time_index(target_time, times)
        actual_time = float(times[k])
        
        x = ds.x.values
        ymid = len(ds.y) // 2
        
        h_num = ds.h[k, ymid, :].values
        zb_num = ds.z_b[k, ymid, :].values
        hu_num = ds.hu[k, ymid, :].values
        
        h_ex, u_ex, hu_ex, zb_ex = analytical_solution(x, actual_time)
        
        # Calculate errors for Depth (h)
        err_h = np.abs(h_num - h_ex)
        L1_h.append(np.mean(err_h))
        L2_h.append(np.sqrt(np.mean(err_h**2)))
        Linf_h.append(np.max(err_h))
        
        # Calculate errors for Bed (zb)
        err_zb = np.abs(zb_num - zb_ex)
        L1_zb.append(np.mean(err_zb))
        L2_zb.append(np.sqrt(np.mean(err_zb**2)))
        Linf_zb.append(np.max(err_zb))
        
        # Calculate errors for Momentum (hu)
        err_hu = np.abs(hu_num - hu_ex)
        L1_hu.append(np.mean(err_hu))
        L2_hu.append(np.sqrt(np.mean(err_hu**2)))
        Linf_hu.append(np.max(err_hu))
        
        valid_dh.append(dh)
        ds.close()
        
    return valid_dh, (L1_h, L2_h, Linf_h), (L1_zb, L2_zb, Linf_zb), (L1_hu, L2_hu, Linf_hu)

def plot_convergence(dh_values, norms_h, norms_zb, norms_hu, outdir="results_convergence"):
    os.makedirs(outdir, exist_ok=True)
    
    L1_h, L2_h, Linf_h = norms_h
    L1_zb, L2_zb, Linf_zb = norms_zb
    L1_hu, L2_hu, Linf_hu = norms_hu
    
    if len(dh_values) < 2:
        print("Not enough data points to plot convergence.")
        return

    # Helper function to add reference slopes
    def add_reference_slopes(ax, x, y, order_1_offset=1.0, order_2_offset=0.5):
        if len(x) < 2: return
        idx = min(1, len(x) - 1)
        x_ref = x[idx]
        y_ref = y[idx]
        
        y_order1 = y_ref * (x / x_ref)**1
        y_order2 = y_ref * (x / x_ref)**2
        
        ax.plot(x, y_order1 * order_1_offset, 'k:', label="O(dh) ref")
        ax.plot(x, y_order2 * order_2_offset, 'k--', label="O(dh^2) ref")

    # Plot for h
    fig, ax = plt.subplots(1, 3, figsize=(21, 6), dpi=150)
    
    dh_arr = np.array(dh_values)
    
    ax[0].loglog(dh_arr, L1_h, 'o-', label="L1 (mean)")
    ax[0].loglog(dh_arr, L2_h, 's-', label="L2 (RMS)")
    ax[0].loglog(dh_arr, Linf_h, '^-', label="Linf (max)")
    add_reference_slopes(ax[0], dh_arr, L2_h, order_1_offset=1.5, order_2_offset=0.5)
    
    ax[0].set_xlabel("Mesh Size (dh)")
    ax[0].set_ylabel("Error Norm")
    ax[0].set_title("Convergence of Depth (h)")
    ax[0].set_xticks(dh_arr)
    ax[0].set_xticklabels([f"{x:g}" for x in dh_arr])
    ax[0].grid(True, which="major", ls="-")
    ax[0].grid(True, which="minor", ls="--", alpha=0.5)
    ax[0].legend()
    # Invert x-axis so smaller dh is on the right
    ax[0].invert_xaxis()

    # Plot for zb
    ax[1].loglog(dh_arr, L1_zb, 'o-', label="L1 (mean)")
    ax[1].loglog(dh_arr, L2_zb, 's-', label="L2 (RMS)")
    ax[1].loglog(dh_arr, Linf_zb, '^-', label="Linf (max)")
    add_reference_slopes(ax[1], dh_arr, L2_zb, order_1_offset=1.5, order_2_offset=0.5)
    
    ax[1].set_xlabel("Mesh Size (dh)")
    ax[1].set_ylabel("Error Norm")
    ax[1].set_title("Convergence of Bed Level (z_b)")
    ax[1].set_xticks(dh_arr)
    ax[1].set_xticklabels([f"{x:g}" for x in dh_arr])
    ax[1].grid(True, which="major", ls="-")
    ax[1].grid(True, which="minor", ls="--", alpha=0.5)
    ax[1].legend()
    ax[1].invert_xaxis()

    # Plot for hu
    ax[2].loglog(dh_arr, L1_hu, 'o-', label="L1 (mean)")
    ax[2].loglog(dh_arr, L2_hu, 's-', label="L2 (RMS)")
    ax[2].loglog(dh_arr, Linf_hu, '^-', label="Linf (max)")
    add_reference_slopes(ax[2], dh_arr, L2_hu, order_1_offset=1.5, order_2_offset=0.5)
    
    ax[2].set_xlabel("Mesh Size (dh)")
    ax[2].set_ylabel("Error Norm")
    ax[2].set_title("Convergence of Momentum (hu)")
    ax[2].set_xticks(dh_arr)
    ax[2].set_xticklabels([f"{x:g}" for x in dh_arr])
    ax[2].grid(True, which="major", ls="-")
    ax[2].grid(True, which="minor", ls="--", alpha=0.5)
    ax[2].legend()
    ax[2].invert_xaxis()

    plt.suptitle("Grid Convergence Study")
    plt.tight_layout()
    
    fname = os.path.join(outdir, "convergence_plot.png")
    plt.savefig(fname, bbox_inches="tight")
    print(f"Saved convergence plot to {fname}")
    plt.close()

if __name__ == "__main__":
    target_time = 10.0
    dh_values_to_check = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    
    print(f"Calculating error norms at t={target_time} ...")
    valid_dh, norms_h, norms_zb, norms_hu = calculate_norms(dh_values_to_check, target_time)
    
    if len(valid_dh) > 0:
        plot_convergence(valid_dh, norms_h, norms_zb, norms_hu)
        
        print("\nConvergence Results Summary for h:")
        print(f"{'dh':<8} | {'L1(h)':<10} | {'L2(h)':<10} | {'Linf(h)':<10}")
        print("-" * 50)
        for i, dh in enumerate(valid_dh):
            print(f"{dh:<8} | {norms_h[0][i]:.2e} | {norms_h[1][i]:.2e} | {norms_h[2][i]:.2e}")

        print("\nConvergence Results Summary for hu:")
        print(f"{'dh':<8} | {'L1(hu)':<10} | {'L2(hu)':<10} | {'Linf(hu)':<10}")
        print("-" * 50)
        for i, dh in enumerate(valid_dh):
            print(f"{dh:<8} | {norms_hu[0][i]:.2e} | {norms_hu[1][i]:.2e} | {norms_hu[2][i]:.2e}")
    else:
        print("No valid output files found. Did you run the simulations first?")
