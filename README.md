# cuQSL

Compute the QSL (Quasi-Separatrix Layer) for magnetic field data using CUDA acceleration.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (NVIDIA recommended)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed

### Project Structure
```bash
cuQSL/
├── src/
│   └── cuQSL/              # Core package
│       ├── __init__.py\\
│       ├── cuQSL_cat.py     # Main calculation module
│       └── cuQSL_cat_scripts.py  # Helper scripts
├── setup.py\\
└── pyproject.toml
```

### Install via pip
```bash
# Install from source (recommended for latest features)
git clone git@github.com:gychen-NJU/CuQSL.git
cd cuQSL
pip install --use-pep517  .
```
### Basic Usage
```python
    >>> import cuQSL
    >>> Bxyz = np.load('Bxyz.npy')
    >>> grid_xyz = [xgrid,ygrid,zgrid]
    >>> solver = cuQSL.cat(Bxyz, grid_xyz) # cartesian data
    >>> points = np.load('points.npy')
    >>> device = ['cuda:0','cuda:1']
    >>> logQ,Length,Twist = solver(points, devices=device)
'''
