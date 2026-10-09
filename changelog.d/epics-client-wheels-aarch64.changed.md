`osprey-connectors` now requires p4p 4.3 or newer, which brings pvxslibs 1.5.3
and epicscorelibs 7.0.10.99.0.2. On Python 3.14 these install from prebuilt
wheels on linux aarch64 too, so an arm64 install there no longer needs a C
compiler; on Python 3.11 to 3.13, arm64 still builds them from source.
