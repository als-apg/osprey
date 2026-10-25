The artifacts gallery now sanitizes agent-written markdown before rendering it.
Raw HTML in that markdown used to run as script on the console's origin; math
still renders.
