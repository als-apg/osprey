The bundled service compose templates render the site's image build args
through one shared Jinja macro instead of a copy of the same loop per
template, so a build arg added to the set reaches every image OSPREY builds.
The bluesky template's publisher block now names the bridge's document-plane
forwarder, and every service template opens by naming the suites that pin its
render byte for byte.
