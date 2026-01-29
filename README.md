# PyExner

Accelerated Hydrodynamic and Morphological Simulations using JAX

[![license](https://img.shields.io/github/license/jkhansell/PyExner?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff)](https://choosealicense.com/licenses/)
[![last-commit](https://img.shields.io/github/last-commit/jkhansell/PyExner?style=default&logo=git&logoColor=white&color=0080ff)](https://github.com/jkhansell/PyExner/commits/main)
[![repo-top-language](https://img.shields.io/github/languages/top/jkhansell/PyExner?style=default&color=0080ff)](https://github.com/jkhansell/PyExner)
[![repo-language-count](https://img.shields.io/github/languages/count/jkhansell/PyExner?style=default&color=0080ff)](https://github.com/jkhansell/PyExner)

## Overview

![](tests/runtime/erodible_channel/fields_with_umag.gif)

PyExner is a Python library for accelerated, parallelized hydrodynamic and morphological simulations, leveraging JAX for high-performance numerical computation. The library is designed to solve shallow water equations, sediment transport, and morphodynamic problems using modern hardware and scalable parallelism.

## Features

- Fast, GPU-accelerated hydrodynamic and Exner equation solvers using JAX.
- Modular architecture for solvers, integrators, domains, and boundaries.
- Utilities for mesh creation, boundary conditions, and diagnostics.
- Support for distributed and multi-GPU computation.
- Ready-to-use example scripts and test cases.
- Extensible framework for research in morphodynamics and sediment transport.

## Project Structure

```
PyExner/
├── LICENSE
├── PyExner/
│   ├── __init__.py
│   ├── config.py
│   ├── domain/
│   ├── integrators/
│   ├── io/
│   ├── runtime/
│   ├── solvers/
│   ├── state/
│   └── utils/
├── PyExner.png
├── README.md
├── pyproject.toml
└── tests/
    ├── domain/
    └── runtime/
```

## Getting Started

### Prerequisites

- Python 3.8+
- [JAX](https://github.com/google/jax)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/) (for visualization)
- [SciPy](https://scipy.org/) (for numerical functions)

### Installation

**Clone the repository:**
```sh
git clone https://github.com/jkhansell/PyExner
cd PyExner
```

**Install dependencies:**
```sh
pip install -r requirements.txt
# or use your preferred environment manager, e.g. mamba or conda
```

**Install PyExner:**
```sh
pip install .
```

### Usage

Basic usage example:
```python
from PyExner import run_driver
params = {
    # Define your simulation parameters here
}
results = run_driver(params)
```
See the [tests/runtime/dambreak.py](https://github.com/jkhansell/PyExner/blob/main/tests/runtime/dambreak.py) script for a full dam-break simulation example.

### Testing

Run the test suite:
```sh
pytest tests/
```

## Contributing

Contributions, suggestions, and bug reports are welcome!

- [Join Discussions](https://github.com/jkhansell/PyExner/discussions)
- [Report Issues](https://github.com/jkhansell/PyExner/issues)
- [Submit Pull Requests](https://github.com/jkhansell/PyExner/pulls)

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- JAX: Accelerated numerical computing
- Contributors and users of PyExner

---

<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="20%">
</p>
