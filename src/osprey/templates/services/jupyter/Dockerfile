FROM jupyter/base-notebook:python-3.11

# Switch to root to install packages
USER root

# Update and install iputils-ping and timezone data
RUN apt-get update && apt-get install -y iputils-ping tzdata && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Switch back to the default jovyan user
USER ${NB_UID}
