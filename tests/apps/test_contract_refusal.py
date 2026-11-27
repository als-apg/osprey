"""The ``contract`` key of ``app.yml``: the number declared, and the refusal of any other."""

from __future__ import annotations

import pytest

from osprey.apps.contract import APP_CONTRACT
from osprey.apps.errors import AppManifestError
from osprey.apps.manifest import parse_manifest
from osprey.apps.origin import AppOrigin
from osprey.docs_links import APP_CONTRACT_URL

from ._contract import CONTRACT, with_contract
from ._installed import DIRECTORY, DIST, MODULE, VERSION

APP = "knob-demo"
BASE = {"name": APP, "skills": [APP]}

N = APP_CONTRACT
UNDECLARED_REMEDY = f'this OSPREY implements contract {N}. Add "contract: {N}" to app.yml.'
SEE = f"See {APP_CONTRACT_URL}."


def _refusal(data: object, source: str | None = None) -> AppManifestError:
    with pytest.raises(AppManifestError) as info:
        parse_manifest(data, app=APP, source=source)
    return info.value


def _only(error: AppManifestError):
    assert len(error.violations) == 1, error.violations
    return error.violations[0]


def test_the_helper_declares_the_current_number() -> None:
    assert CONTRACT == APP_CONTRACT
    assert with_contract(BASE) == {"contract": APP_CONTRACT, **BASE}


def test_the_contract_number_is_a_plain_integer() -> None:
    assert type(APP_CONTRACT) is int
    assert APP_CONTRACT >= 1


def test_the_current_number_passes() -> None:
    manifest = parse_manifest(with_contract(BASE), app=APP)
    assert manifest.name == APP
    assert manifest.contract == APP_CONTRACT


def test_a_missing_contract_is_refused() -> None:
    violation = _only(_refusal(dict(BASE)))
    assert (violation.app, violation.field, violation.rule) == (APP, "contract", "contract")
    assert violation.message == f"not declared; {UNDECLARED_REMEDY} {SEE}"
    assert str(violation).startswith(f"apps/{APP}/app.yml: contract: not declared;")


@pytest.mark.parametrize(
    ("value", "shown"),
    [(True, "True"), ("1", "'1'"), (1.0, "1.0"), (None, "None"), ([1], "[1]")],
    ids=["true", "string", "float", "null", "list"],
)
def test_a_contract_that_is_not_an_integer_is_refused(value: object, shown: str) -> None:
    violation = _only(_refusal({**BASE, "contract": value}))
    assert violation.field == "contract"
    assert violation.rule == "contract"
    assert violation.message == f"{shown} is not an integer; {UNDECLARED_REMEDY} {SEE}"


def test_a_lower_contract_is_refused_asking_for_an_update() -> None:
    lower = N - 1
    violation = _only(_refusal({**BASE, "contract": lower}))
    assert violation.message == (
        f"the app is written for contract {lower}; this OSPREY implements {N}. "
        f"Update the app to contract {N}. {SEE}"
    )


def test_a_higher_contract_is_refused_asking_for_an_upgrade() -> None:
    higher = N + 1
    violation = _only(_refusal({**BASE, "contract": higher}))
    assert violation.message == (
        f"the app is written for contract {higher}; this OSPREY implements {N}. "
        f"Upgrade OSPREY. {SEE}"
    )


def test_a_mismatch_is_reported_alone_beside_an_unknown_key() -> None:
    error = _refusal({**BASE, "contract": N + 1, "no_such_key": 1})
    violation = _only(error)
    assert violation.field == "contract"
    assert "no_such_key" not in str(error)


def test_a_missing_contract_is_reported_alone_beside_an_unknown_key() -> None:
    violation = _only(_refusal({**BASE, "no_such_key": 1}))
    assert violation.field == "contract"


def test_a_document_that_is_not_a_mapping_keeps_the_mapping_refusal() -> None:
    violation = _only(_refusal(["contract", N]))
    assert violation.rule == "mapping"


def test_an_installed_app_refusal_names_distribution_and_version_once() -> None:
    origin = AppOrigin("installed", DIRECTORY, distribution=DIST, version=VERSION, module=MODULE)
    source = origin.manifest_source
    assert source is not None
    error = _refusal({**BASE, "contract": N + 1}, source=source)
    line = str(_only(error))
    assert line.startswith(f"{source}: contract: ")
    assert line.count(f"{DIST} {VERSION}") == 1
    assert str(error).count(f"{DIST} {VERSION}") == 1
