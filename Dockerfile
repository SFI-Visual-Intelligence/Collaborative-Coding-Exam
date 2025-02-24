FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:debian

# Set a temporary working directory (will be switched in the kubectl job anyways...)
WORKDIR /temp

# Copy the pyproject file to the working directory to ensure exact versions
COPY pyproject.toml .

# Make a empty venv
RUN uv venv

# Install all dependencies from the pyproject file
RUN uv add pyproject.toml

# Set the path to the venv
ENV PATH="/temp/.venv/bin:${PATH}"
