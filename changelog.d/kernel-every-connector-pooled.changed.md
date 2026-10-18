A notebook cannot import a control-system client library (epics, p4p, caproto,
aioca, pvaccess, tango, doocs4py, epicscorelibs) or load one through ctypes or
cffi, for reads as for writes; ophyd v1 and `ophyd_async.epics` are refused
with them, so PVAccess puts, rpc, Tango commands and serving a PV from a cell
are no longer possible in a kernel. Every kernel, on a single-target or
in-process simulator deployment too, reads and writes through a connector-host
child it starts on its first control-system call.
