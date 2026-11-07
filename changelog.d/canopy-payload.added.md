The CANOPY panel ships as a prebuilt payload: the `canopy` and
`pydmconverter` wheels, the viewer (whose `ir-map.json` is the widget
catalog), Trace and, when it has one, the staged spec of screen-format and
widget property schemas, made by the canopy repo's
`scripts/build-payload.sh`. The `canopy-payload` pre-build step
(`python -m osprey.integrations.canopy.stage`) copies the payload
`OSPREY_CANOPY_PAYLOAD` names into the deployment repository, and refuses one
whose `STAGED.json` and `spec/` disagree or whose commits are not the CANOPY
and PyDMConverter commits this OSPREY is proven against.
`python -m osprey.integrations.canopy` reports whether the installed CANOPY is
at those commits (only a git install at the pin counts), and `--pins` prints
them. `OSPREY_CANOPY_PAYLOAD_CHECK=0` skips staging and the pin check, for
development builds only. The image build checks a deployment's CANOPY screen
library and facility collections with
`python -I -m osprey.integrations.canopy.image_check library|collections`; an
image without CANOPY passes both checks and says so in the build log.
