`config: {images.platform: linux/amd64}` in `profile.yml` names the platform
every image OSPREY builds is built for: `osprey build` judges deployment wheels
against it and names it on the project Dockerfile's `FROM`, and `osprey up`
builds every image and compose-built service for it. Unset, images are built
for this host, as before.
