FROM ghcr.io/open-webui/pipelines:main

# Copy files that are needed to run Osprey
RUN mkdir -p /pipelines
COPY . /pipelines

# Accept build arguments that control which version of Osprey is used
ARG DEV_MODE

# Install the Osprey framework
RUN /bin/bash /pipelines/start.sh
RUN rm /pipelines/start.sh  # The installation script is no longer needed