The qmd sidecar image carries three patches on the pinned qmd 2.5.3 release,
each proposed upstream: a natural-language question now matches on its content
words instead of requiring every word, and vector search scans an in-memory
copy of the vectors. Searches on a large index drop from seconds to about a
quarter of a second; the cost is about 3 KiB of RAM per indexed chunk per
sidecar, and `QMD_VEC_SCAN=vec0` turns the in-memory scan off.
