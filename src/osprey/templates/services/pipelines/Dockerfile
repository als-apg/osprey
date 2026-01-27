FROM ghcr.io/open-webui/pipelines:main

# Copy files that are needed to run Osprey
RUN mkdir -p /pipelines
COPY . /pipelines

# Define environment variables that control which version of Osprey is used
ENV DEV_MODE=${DEV_MODE}
ENV OSPREY_WHEEL=${OSPREY_WHEEL}
ENV PIPELINES_DIR=${PIPELINES_DIR}

# Install the Osprey framework
RUN /bin/bash /pipelines/start.sh
RUN rm /pipelines/start.sh  # The installation script is no longer needed