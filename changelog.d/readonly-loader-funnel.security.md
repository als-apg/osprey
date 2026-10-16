A readonly Python run now refuses shared-library loads at the loader underneath
every `ctypes` spelling and at cffi's `FFI.dlopen`, closing `_ctypes.dlopen`,
cffi and `ctypes.CDLL` references bound before the guard was installed. The
pyepics read path is unchanged.
