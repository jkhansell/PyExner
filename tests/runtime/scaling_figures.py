import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GPUs = [i for i in np.logspace(0,6,7,base=2)]

data = pd.read_csv("elapsed_times.txt", delimiter=":", header=None)
data["GPUs"] = GPUs
data["speedup"] = data[1][0]/data[1] 

print(data)

plt.figure(figsize=(8,6))
plt.plot(data["GPUs"], data[1], marker=".")
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Number of GPUs")
plt.ylabel("Execution time [s]")
plt.grid(alpha=0.25,which='both')
# Set x-axis ticks at powers of 2 from 1 to 64
xticks = [2**i for i in range(0, 7)]  # [1, 2, 4, 8, 16, 32, 64]
plt.xticks(xticks, labels=[str(x) for x in xticks])
plt.savefig("executiontimes.png",dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(data["GPUs"], data["speedup"], marker=".")
plt.plot(data["GPUs"], data["GPUs"], linestyle="dashed", c="red")
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Number of GPUs")
plt.ylabel("Speedup")
plt.grid(alpha=0.25,which='both')
# Set x-axis ticks at powers of 2 from 1 to 64
xticks = [2**i for i in range(0, 7)]  # [1, 2, 4, 8, 16, 32, 64]
plt.xticks(xticks, labels=[str(x) for x in xticks])
plt.savefig("speedup.png",dpi=200)
plt.close()
