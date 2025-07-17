# 🌊 PyExner

**PyExner** is a high-performance, distributed 2D solver for the **Shallow Water Equations (SWE)** coupled with the **Exner equation** for sediment transport. Built on [JAX](https://github.com/google/jax), it supports **GPU acceleration**, **multi-host parallelism**, and **automatic differentiation** for research in geomorphology, sediment dynamics, and flood modeling.

---

## 🚀 Features

- 🧮 Numerical schemes: Roe, HLL, SWE–Exner, SWE–Richardson coupling
- 🧠 Auto-differentiable and JIT-compiled using JAX
- 🔁 Parallelism via `pjit` and 2D domain decomposition
- ⚡ Native support for GPU (NVIDIA/AMD) clusters and SLURM
- 🧱 Halo exchange with periodic or physical boundary conditions
- 📊 Diagnostic tools (error norms, convergence tracking)
- 📦 Modular design for experiments and model extensions

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/PyExner.git
cd PyExner

``` 

### 2. 

```bash
git clone https://github.com/yourusername/PyExner.git
cd PyExner
