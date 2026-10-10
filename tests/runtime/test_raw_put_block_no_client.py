"""The ``no_client`` mode of the raw-put block: no control-system client at all.

``install("no_client", ...)`` refuses a client package at import and a client
shared library at load, for reads and writes alike. Every in-process contract
here names fake packages, fake libraries and fake symbols, never the real
ones, so the test process itself never loses a client; the cases that need a
real client library run in a subprocess. The fixture from
:mod:`tests.runtime._patch_restore` puts back every loader the block replaced
and drops its finders.
"""

from __future__ import annotations

from osprey_connectors.errors import ClientLibraryRefusedError


def test_refusal_is_a_runtime_error_not_an_import_error():
    error = ClientLibraryRefusedError(package="epics")
    assert isinstance(error, RuntimeError)
    assert not isinstance(error, ImportError)
    assert error.reason == "RAW_CLIENT_IMPORT"
    assert error.package == "epics"
    assert error.library is None


def test_refusal_message_names_the_runtime_route():
    by_package = str(ClientLibraryRefusedError(package="p4p"))
    by_library = str(ClientLibraryRefusedError(library="libca.so.7"))
    for message in (by_package, by_library):
        assert message.startswith(ClientLibraryRefusedError.MARKER)
        assert "osprey.runtime" in message
        for call in ("read_channel(address)", "write_channel(address, value)", "write_channels("):
            assert call in message
    assert "'p4p'" in by_package
    assert "'libca.so.7'" in by_library
    assert "client library" in by_library
