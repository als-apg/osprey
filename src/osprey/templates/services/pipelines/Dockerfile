FROM ghcr.io/open-webui/pipelines:main

RUN mkdir -p /pipelines
COPY . /pipelines
RUN /bin/bash /pipelines/start.sh  # Installs the Osprey framework
RUN rm /pipelines/start.sh  # The installation script is no longer needed