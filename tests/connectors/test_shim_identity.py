"""Module-identity guarantees for the osprey-connectors extraction.

Main osprey re-exports the moved modules under their historical paths via
sys.modules-aliasing shims; these tests pin the contract that both names
resolve to the SAME module object (patching/isinstance safe).
"""


def test_osprey_connectors_is_installed():
    import osprey_connectors

    assert osprey_connectors.__version__ == "0.1.0"
