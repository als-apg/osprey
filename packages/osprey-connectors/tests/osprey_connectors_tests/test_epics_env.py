"""The inherited EPICS client environment: one prefix list, a scrub and a copy without it."""

from osprey_connectors.epics_env import EPICS_ENV_PREFIXES, scrub_epics_env, without_epics_env

_ENV = {
    "EPICS_CA_ADDR_LIST": "gw.example",
    "EPICS_CA_AUTO_ADDR_LIST": "NO",
    "EPICS_PVA_NAME_SERVERS": "gw.example:5075",
    "EPICS_PVAS_SERVER_PORT": "5075",
    "EPICS_BASE": "/opt/epics",
    "PATH": "/usr/bin",
}


def test_the_prefixes_are_the_two_client_families():
    assert EPICS_ENV_PREFIXES == ("EPICS_CA_", "EPICS_PVA_")


def test_the_scrub_removes_the_client_variables_in_place_and_says_which():
    env = dict(_ENV)

    removed = scrub_epics_env(env)

    assert removed == {
        "EPICS_CA_ADDR_LIST": "gw.example",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
        "EPICS_PVA_NAME_SERVERS": "gw.example:5075",
    }
    assert env == {"EPICS_PVAS_SERVER_PORT": "5075", "EPICS_BASE": "/opt/epics", "PATH": "/usr/bin"}


def test_the_copy_leaves_the_same_variables_out_and_the_original_alone():
    copy = without_epics_env(_ENV)

    assert copy == {
        "EPICS_PVAS_SERVER_PORT": "5075",
        "EPICS_BASE": "/opt/epics",
        "PATH": "/usr/bin",
    }
    assert "EPICS_CA_ADDR_LIST" in _ENV
