# Installation Guideline

This guide covers the recommended installation procedure for SAOS, targeting a Linux environment (e.g., Ubuntu 24). 
SAOS requires Python >= 3.10. Since it propagates the light paths in threads, a **free-threaded Python** (no GIL) is highly
recommended: with the GIL enabled the results are the same but the light paths do not run in parallel (SAOS logs a notice at
import). Tested configuration: Python 3.13.9t, torch 2.11 (CUDA 13.0), h5py 3.16, pyzmq 27.1, pvlib 0.15.

> **Note on free-threading:** torch support for 3.13t is experimental and PyTorch is moving its free-threaded wheels to 3.14t.
> `PYTHON_GIL=0` also forces extensions that do not declare free-threading support (OpenCV, astropy, pyzmq) to run without the
> GIL.

---

## 1. Prerequisites (Ubuntu)

Install the necessary system dependencies required to build Python and other scientific packages:

```bash
sudo apt update
sudo apt install -y build-essential curl git libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

## 2. Python Environment Setup

We recommend using `pyenv` to manage your Python versions and virtual environments.

### Install `pyenv` and `pyenv-virtualenv`

Follow the official [pyenv installation instructions](https://github.com/pyenv/pyenv) or run the installer:

```bash
curl https://pyenv.run | bash
```
Make sure to add `pyenv` to your `~/.bashrc` or `~/.profile` as instructed by the installer.

### Install Python 3.13 (Free-Threaded)

Install the free-threaded version of Python 3.13 (or newer) and create a virtual environment:

```bash
pyenv install 3.13.9t
pyenv virtualenv 3.13.9t venv_saos
pyenv activate venv_saos
```

To optimize the Python execution, ensure you upgrade `pip` and disable the limited API for CMake packages:

```bash
pip install --upgrade pip
export CMAKE_ARGS="-DPYTHON3_LIMITED_API=OFF"
```

> **Note:** You can add `export PYTHON_GIL=0` to your `~/.bashrc` to ensure the Global Interpreter Lock is disabled when running the free-threaded version.

---

## 3. Clone and Install SAOS

Clone the SAOS repository and install it in editable mode (`-e`) so you can modify the source code easily.

```bash
mkdir -p ~/projects
cd ~/projects

# Clone the repository
git clone https://github.com/nrodlin/SAOS.git
cd SAOS

# Checkout to the development branch (if required)
git checkout develop

# Install SAOS (editable) with its runtime dependencies
pip install --prefer-binary -e .

# Optional: documentation (mkdocs) and validation-suite tools (pytest, pytest-timeout, pyyaml, psutil, ruff, jupyter)
pip install --prefer-binary -r requirements.txt
```

The validation suite is described in `ci/README.md` (`ci/run_ci.sh fast`, or `python -m pytest tests -m fast`). `torch` runs on the CPU by default; a CUDA build of torch is optional and is used automatically when `torch.cuda.is_available()`.

> **Free-threaded builds and wheels:** use `--prefer-binary`. On a free-threaded Python the newest release of several
> dependencies is often published only as source (e.g. scipy 1.18, matplotlib 3.11.2, pyzmq 27.2 at the time of writing),
> and pip would try to compile it (scipy then fails without a Fortran compiler). `--prefer-binary` picks the newest version
> with a free-threaded wheel instead (checked on 2026-10-27 for 3.13t: numpy 2.4, scipy 1.17.1, matplotlib 3.11.1,
> pyzmq 27.1.0, torch 2.12).
>
> **Packages built from source:** `opencv-contrib-python` and `h5py` have no free-threaded wheels, so pip compiles them.
> OpenCV is the long one (of the order of one hour on a 12-core laptop); it needs the build tools of section 1, `cmake`,
> and the `CMAKE_ARGS="-DPYTHON3_LIMITED_API=OFF"` exported in section 2. For h5py, see the optional HDF5 build below.
> The compiled wheels stay in pip's cache, so later environments install them in seconds.

The runtime dependencies (h5py, joblib, matplotlib, numpy, opencv-contrib-python, pvlib, pyzmq, scipy, torch) are declared in `setup.cfg` and installed by pip. A non-editable install (`pip install .` or a wheel built with
`pip wheel . --no-deps`) contains every subpackage and the solar images (~90 MB compressed), so SAOS also works outside the
repository.

### Git Large File Storage (LFS)

SAOS uses **Git LFS** to store large binary assets such as solar granulation images
(e.g., `SAOS/images/cont_500nm.h5`, `cont_680nm.h5`). These files are required for
solar AO simulations with `ExtendedSource`. Without pulling them, the simulator
will fail to load the solar image data.

**1. Install git-lfs** (if not already installed):

```bash
# Ubuntu / Debian
sudo apt install git-lfs

# macOS (Homebrew)
brew install git-lfs

# Or via conda
conda install -c conda-forge git-lfs
```

**2. Enable LFS in your git configuration** (one-time setup):

```bash
git lfs install
```

**3. Pull the LFS-tracked files** after cloning:

```bash
# Inside the SAOS directory
git lfs pull
```

This will download the solar image files. You can verify them with:

```bash
git lfs ls-files
```

> **Note:** `git lfs pull` requires network access to GitHub. If you are on a
> restricted network, ask your system administrator to mirror the LFS objects.

## 4. Additional Dependencies

h5py (data files, M7 discrete models) and pyzmq (telemetry streaming) are installed with SAOS.

### Compiling HDF5 and h5py from source (Optional)

If you encounter issues with pre-compiled `h5py` binaries, you can compile HDF5 and `h5py` from source:

```bash
# 1. Compile HDF5
git clone https://github.com/HDFGroup/hdf5.git
cd hdf5
git checkout hdf5-1_14_3
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/hdf5-install -DHDF5_ENABLE_THREADSAFE=ON -DHDF5_BUILD_CPP_LIB=OFF -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
make install
cd ../..

# 2. Compile h5py against the custom HDF5
git clone https://github.com/h5py/h5py.git
cd h5py
export HDF5_DIR=$HOME/hdf5-install
pip install .
```

---

You are now ready to run your first simulation! Head over to the [Tutorials](tutorials/01_intro_to_saos.ipynb) section to learn the basics of SAOS.
