FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS builder

# Install the project into `/pyfhd`
WORKDIR /pyfhd

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Disable Python downloads, because we want to use the system interpreter
# across both images. If using a managed Python version, it needs to be
# copied from the build image into the final image; see `standalone.Dockerfile`
# for an example.
ENV UV_PYTHON_DOWNLOADS=0

# Copy everything over except what's in the .dockerignore
COPY . /pyfhd

ENV XDG_CACHE_HOME="/pyfhd/.cache"

RUN mkdir -p /pyfhd/.cache/astropy /pyfhd/.config

# Install the dependencies and project (This also creates the __git__.py file)
RUN uv sync --frozen --no-dev

# Remove the cache directory for uv to reduce image size
RUN rm -rf /pyfhd/.cache/uv

# Update the astropy caches
RUN .venv/bin/python -c "from astropy.utils.iers import IERS_Auto; iers_a = IERS_Auto.open()"
RUN .venv/bin/python -c "from astropy.coordinates import EarthLocation; site_names = EarthLocation.get_site_names()"

# Grab the MWA FEE Beam file
RUN cd /pyfhd/src/pyfhd/resources/instrument_config && wget "http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5" && cd /pyfhd

RUN rm -rf /pyfhd/.git /pyfhd/.cache/uv 

# Separate build stage as uv is not needed in the final image
FROM python:3.13-slim-bookworm AS runner

# This essentially "installs" pyfhd into the final image
COPY --from=builder --chown=1000:1000 /pyfhd /pyfhd

# Set the cache for astropy
ENV XDG_CACHE_HOME="/pyfhd/.cache"

# Set the path to the virtual environment so it can find pyfhd
ENV PATH="/pyfhd/.venv/bin:$PATH"

WORKDIR /pyfhd
