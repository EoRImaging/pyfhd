# Installation

The dependencies on pyfhd to run with FHD have been removed, this makes installing
and `pyfhd` much easier than FHD. `pyfhd` is currently supported for Python 3.11+.

## Installing to do some development on pyfhd?

Please follow the [contribution guide](../develop/contribution_guide.md).

## pip

pyfhd is on [PyPi](https://pypi.org/) and is installable via pip. Typically
you'll create a virtual environment using `venv`, `conda` or `uv` to ensure
you're not installing things into any other python installations you may have.
Typically `venv` is best in places where you are restricted by what you can and
can't install (HPC environments). `uv` is best used for development as it supports
a lot of features that Python developers of packages will appreciate (oh,
and by the way, _it's fast_, like _really fast_ at installing packages).
`conda` is best used where you need to use more than just Python, but perhaps a
combination of languages and tools, `conda` has the ability to manage any
compilers, CUDA versions, system tools etc.

### Make a venv environment
This assumes you have python installed.

```bash
# Make a virtual environment in the current directory, using the name pyfhd
python -m venv pyfhd
# Activate the environment
source pyfhd/bin/activate
```

You should now notice some brackets `(pyfhd)` present in the terminal. You can
deactivate the environment at any time by putting `deactivate` in the terminal.

### Make a venv via uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv
source .venv/bin/activate
```

Since `uv` just uses `venv` under the hood, you can get out of the environment
by calling `deactivate` at any time.

### Make a conda environment

Install [conda](https://github.com/conda-forge/miniforge#install)

```bash
conda env create --file environment.yml python=3.11 # You can make it 3.11 or 3.12 or 3.13
conda activate pyfhd
```

You can deactivate the environment anytime by using `conda deactivate`

### Installing pyfhd

Inside your virtual environment, run

```bash
pip install pyfhd
```

You can Verify the installation using the version command

```bash
pyfhd -v
```

It should give you output that looks like this:

```
    ________________________________________________________________________
    |    ooooooooo.               oooooooooooo ooooo   ooooo oooooooooo.    |
    |    8888   `Y88.             8888       8 8888    888   888     Y8b    |
    |    888   .d88' oooo    ooo  888          888     888   888      888   |
    |    888ooo88P'   `88.  .8'   888oooo8     888ooooo888   888      888   |
    |    888           `88..8'    888          888     888   888      888   |
    |    888            `888'     888          888     888   888     d88'   |
    |    o888o            .8'     o888o        o888o   o888o o888bood8P'    |
    |                 .o..P'                                                |
    |                `Y8P'                                                  |
    |_______________________________________________________________________|

    Python Fast Holographic Deconvolution

    Translated from IDL to Python as a collaboration between Astronomy Data and
    Computing Services (ADACS) and the Epoch of Reionisation (EoR) Team.

    Repository: https://github.com/EoRImaging/pyfhd

    Documentation: https://pyfhd.readthedocs.io/en/latest/

    Version: 1.0

    Git Commit Hash: 176fa3aaf9aeca44b38f00ac4745d9b3a9eefe9c (tutorial_adjustments)
```
