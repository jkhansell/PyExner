import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys 

if __name__ == "__main__":

    data_path = sys.argv[1]
    dataset = xr.open_dataset(data_path)

    timesteps = [0.0, 0.1, 0.3, 0.6]
    total_time = 1.0
    dt = 0.1
    N_half = dataset.h.shape[1] // 2

    time_array = np.arange(0, total_time + dt, dt)

    for i, idt in enumerate(time_array):
        print(i, idt)
        if np.any(np.isclose(idt, timesteps, atol=1e-8)):
            # do something
            x = dataset.x
            h = dataset.h[i, N_half]
            print(dataset.h)

            z = dataset.z[i, N_half]

            fig = plt.figure(figsize=(10,6))
            fig.suptitle(f"Time: {idt} s")
            gs = fig.add_gridspec(2,1, height_ratios=[1.8, 1])
            
            ax_1 = fig.add_subplot(gs[0])
            ax_2 = fig.add_subplot(gs[1])

            ax_1.plot(x, h + z, label=r"sim-$h$", c="black", marker='o', markerfacecolor='none', markeredgecolor="black",markersize=2)
            ax_1.set_ylabel(r"Water level $h$ [m]")

            ax_2.plot(x, z , label=r"sim-$z$", c="black", marker='o', markerfacecolor='none', markeredgecolor="black",markersize=2)
            ax_2.set_ylabel(r"Bed height $z$ [m]")
            ax_2.set_xlabel(r"Channel position $x$ [m]")

            if "subcritical" in data_path:
                if np.isclose(idt,0.1):
                    print("here")
                    ax_1.set_xlim(-1, 1)
                    ax_2.set_xlim(-1, 1)
                    ax_1.set_ylim(1.15, 2.05)
                    ax_2.set_ylim(0.9, 1.05)

                elif np.isclose(idt,0.3):
                    print("here")
                    
                    ax_1.set_xlim(-1.65, 1.65)
                    ax_2.set_xlim(-1.65, 1.65)
                    ax_1.set_ylim(1.15, 1.8)
                    ax_2.set_ylim(0.9, 1.05)

                elif np.isclose(idt,0.6):
                    print("here")

                    ax_1.set_xlim(-2.65, 2.65)
                    ax_2.set_xlim(-2.65, 2.65)
                    ax_1.set_ylim(1.15, 1.8)
                    ax_2.set_ylim(0.9, 1.05)

                ax_1.grid(alpha=0.25)
                ax_2.grid(alpha=0.25)
        
            fig.savefig(f"dambreak_{np.round(idt, decimals=2)}.png", dpi=300)
            plt.close()

        plt.imshow(dataset.h[i])
        plt.savefig(f"h_{i}.png")
        plt.close()

        plt.imshow(dataset.hu[i])
        plt.savefig(f"hu_{i}.png")
        plt.close()

        plt.imshow(dataset.hv[i])
        plt.savefig(f"hv_{i}.png")
        plt.close()

        plt.imshow(dataset.z[i])
        plt.savefig(f"z_{i}.png")
        plt.close()
            


    