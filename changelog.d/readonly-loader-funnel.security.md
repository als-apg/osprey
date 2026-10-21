A readonly Python run refuses every shared-library load at the loader underneath
every `ctypes` spelling and at cffi's `FFI.dlopen`, closing `_ctypes.dlopen`,
cffi and `ctypes.CDLL` references bound before the guard was installed.
