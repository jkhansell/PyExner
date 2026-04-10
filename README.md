# PyExner

**High-performance 2D Shallow Water Equation (SWE) and Exner solver with JAX acceleration and distributed GPU support.**

[![license](https://img.shields.io/github/license/jkhansell/PyExner?style=flat-square&logo=opensourceinitiative&logoColor=white&color=0080ff)](https://choosealicense.com/licenses/bsd-3-clause/)
[![last-commit](https://img.shields.io/github/last-commit/jkhansell/PyExner?style=flat-square&logo=git&logoColor=white&color=0080ff)](https://github.com/jkhansell/PyExner/commits/main)
[![repo-top-language](https://img.shields.io/github/languages/top/jkhansell/PyExner?style=flat-square&color=0080ff)](https://github.com/jkhansell/PyExner)

---

## 🌊 Overview

PyExner is a Python-based numerical engine designed for accelerated, parallelized hydrodynamic and morphodynamic simulations. By leveraging **JAX**, it provides high-performance computation on both CPUs and GPUs (NVIDIA/AMD) with minimal code changes. The library supports distributed memory parallelism via **MPI**, enabling large-scale simulations across multiple GPU nodes.

Its primary application is solving the coupled systems of shallow water equations and sediment transport (Exner equations) to model riverbed evolution, dam-breaks, and complex channel flows.

<p align="center">
  <img src="tests/runtime/erodible_channel/fields_with_umag.gif" width="80%" alt="Erodible Channel Simulation">
</p>

## ✨ Key Features

- **🚀 Performance**: Native JAX implementation for XLA-optimized kernels (JIT compilation).
- **🖥️ Hardware Agnostic**: Run the same code on CPU, NVIDIA GPUs (CUDA), or AMD GPUs (ROCm).
- **🌐 Scaling**: Distributed computing via MPI for multi-GPU and multi-node clusters.
- **📚 Physics-Rich**: 
    - 2D Shallow Water Equations (SWE).
    - Coupled Exner equations for bed evolution.
    - Multiple flux schemes: **Roe**, **Roe-Exner**, and **HLLC**.
- **🛠️ Modular Architecture**:
    - Extensible solvers and time integrators (**Forward Euler**, **SSPRK2**).
    - Flexible boundary condition management (Reflective, Transmissive, etc.).
    - Robust I/O using **PnetCDF** for high-performance state reading/writing.

## 📦 Installation

### Prerequisites

- Python 3.9+
- [MPI](https://www.open-mpi.org/) (for distributed support)

### 1. Clone the Repository
```bash
git clone https://github.com/jkhansell/PyExner
cd PyExner
```

### 2. Install Dependencies
PyExner uses optional dependencies to tailor JAX to your hardware:

**For CPU:**
```bash
pip install ".[cpu]"
```

**For NVIDIA GPU (CUDA 12):**
```bash
pip install ".[cuda12]"
```

### 3. Developer Install
If you plan to modify the code:
```bash
pip install -e ".[dev]"
```

### PnetCDF (Parallel NetCDF)
Building the PnetCDF C library and its Python wrapper is mandatory for high-performance I/O:

1. **Build the C Library**: Download and install [Parallel NetCDF](https://parallel-netcdf.github.io/wiki/Download.html).
2. **Set Environment Variables**:
   ```bash
   export PNETCDF_DIR=/path/to/pnetcdf/installation
   export CC=/path/to/mpicc
   ```
3. **Install Python Wrapper**:
   ```bash
   pip install pnetcdf
   # Or from source if needed:
   # git clone https://github.com/pnetcdf/pnetcdf-python
   # cd pnetcdf-python && python setup.py install
   ```

### mpi4jax
`mpi4jax` enables zero-copy MPI communication for JAX arrays. Install it after the main package:

```bash
CUDA_ROOT=XXX pip install mpi4jax
```
If you are using CUDA-aware MPI, ensure you set the following environment variable at runtime:
```bash
export MPI4JAX_USE_CUDA_MPI=1
```

## 🚀 Quickstart

The easiest way to run a simulation is using the provided `driver`:

```python
from PyExner.runtime.driver import run_driver

# Run a simulation using a configuration file
state, (X, Y) = run_driver("path/to/your/input.yaml")

print(f"Simulation completed. Final state shape: {state.h.shape}")
```

## 🧪 Runtime Examples & Validation

The `tests/runtime/` directory contains several benchmark cases used for physics validation and as reference templates for new simulations.

### 🌊 Hydrodynamics (SWE)
- **1D/2D Dam-break**: Classic benchmarks for shallow water solvers. These cases compare numerical results against analytical solutions (Stoker's solution).
- **Domain Stress Tests**: `L_domain`, `T_domain`, and `Square_domain` test complex geometries and boundary decomposition.

### ⏳ Morphodynamics (Exner)
- **Erodible Channel**: A high-fidelity validation case modeling bed evolution in a channel with specific bottlenecks. This case is calibrated against experimental data and demonstrates the coupling between SWE and Exner equations.

### 🏃 How to Run
Most examples can be executed by running their respective Python scripts or through the driver:
```bash
cd tests/runtime/1D_dambreak
python dambreak.py
```
For distributed runs using MPI:
```bash
mpirun -n 4 python dambreak.py
```

## ⚙️ Configuration

Simulations are controlled via YAML configuration files. Below is a comprehensive example showing all available settings:

```yaml
# 🌐 Global & Solver Settings
end_time: 20.0       # Total simulation time (float)
out_freq: 0.25       # File output interval (float)
cfl: 0.5             # Courant–Friedrichs–Lewy safety number
flux_scheme: Roe     # Numerical scheme: Roe, Roe Exner, or HLLC
integrator: SSPRK2   # Time-stepping: Forward Euler or SSPRK2

# 🌐 Parallelism & I/O
parNx: 4             # MPI partitions in X
parNy: 4             # MPI partitions in Y
input_file: start.nc # Initial condition NetCDF
output_file: out.nc  # Results output path

# ⏳ Erosion & Morphodynamics (Required for Roe Exner)
erosion:
  grass_factor: MPM    # Model (MPM) or constant value
  bulk_porosity: 0.42  # Bed porosity
  sediments:
    sed1:
      fraction: 1.0
      diameter: 1.61e-3
      density: 2630
      erosion_flux: 0.04
      deposition_flux: 0.01

# 🧱 Boundaries
boundaries:
  my_boundary_name:
    type: Reflective                   # Reflective, Transmissive, or Periodic
    polygon: [[0, 0], [1, 0], [1, 1], [0, 1]] 
    values: [NaN, 0.0, NaN, NaN]      # Boundary values [h, u, v, z]
    normal: [-1.0, 0.0]               # Normal vector pointing OUT
```

## 📂 Project Structure

```text
PyExner/
├── src/PyExner/
│   ├── domain/      # Mesh and Boundary management
│   ├── integrators/ # Time-stepping schemes (SSPRK2, Euler)
│   ├── io/          # PnetCDF readers and visualizers
│   ├── parallel/    # MPI utilities
│   ├── runtime/     # Driver logic
│   ├── solvers/     # Flux kernels (Roe, HLLC, Exner)
│   └── state/       # Simulation state definitions
├── tests/
│   ├── runtime/     # Example simulation cases (Dam-break, Erodible Channel)
│   └── domain/      # Unit tests for domain logic
└── pyproject.toml   # Project metadata and dependencies
```

## 📈 Performance & Scaling

PyExner is designed for high-throughput research. Significant speedups are achieved when transitioning from CPU-based solvers to GPU-parallelized JAX kernels.

- **Intra-node**: JAX handles local GPU acceleration.
- **Inter-node**: MPI partitions the domain across nodes.

See `tests/runtime/scaling_figures.py` for scripts to benchmark your specific hardware.

## 🤝 Contributing

We welcome contributions! Please feel free to:
1. Fork the repo.
2. Create a feature branch.
3. Submit a Pull Request.

For bugs or feature requests, please use the [Issue Tracker](https://github.com/jkhansell/PyExner/issues).

## 📄 License

PyExner is released under the **BSD 3-Clause License**. See the [LICENSE](LICENSE) file for details.

---
*Developed with ❤️ for the computational hydrology community.*
