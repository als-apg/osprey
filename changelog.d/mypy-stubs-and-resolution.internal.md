The type check now runs over the source tree instead of stopping before it
examines anything: the stub packages its imports need are declared, a
dependency whose syntax the targeted Python cannot parse is no longer
followed, and both the framework source and the connectors package resolve
from source however the check is invoked.
