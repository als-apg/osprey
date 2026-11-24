`osprey build` is faster for a deployment that mounts a large facility
description. The facility file is indexed once for all the build's renders
rather than once in each, a file of a view is written once for every render
that agrees on what the file itself follows from, and the manifest lists the
digests the build already took of the files it wrote instead of reading them
again. The rendered output is unchanged.
