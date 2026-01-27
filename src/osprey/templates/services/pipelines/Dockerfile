FROM ghcr.io/open-webui/pipelines:main

# First install the latest PyPI release of Osprey and its dependencies
# The intention is that this will be cached between different builds
RUN pip install osprey-framework

# Copy files that are needed to run this particular Osprey agent
RUN mkdir -p /pipelines
COPY . /pipelines

# Potentially install a custom version of the Osprey framework
# Accept build arguments that control which version of Osprey is used
ARG DEV_MODE
RUN /bin/bash /pipelines/start.sh
RUN rm /pipelines/start.sh  # The installation script is no longer needed