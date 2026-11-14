Image builds no longer hold setuptools below 84. They hold `setuptools_dso` at
2.12.4 or later instead, the release that fixed source builds of the EPICS
toolchain (linux/arm64) under setuptools 84.
