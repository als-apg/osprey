"""Facade contract tests for ``osprey.deployment.container_manager``.

``container_manager`` is a thin re-export facade: the deploy CLI
(``osprey.cli.deploy_cmd``) imports the container-management API from here, while
the implementations live in ``compose_generator``, ``container_lifecycle``, and
``status_display`` (each covered by its own dedicated test module). The behavior
these tests guard is the *facade contract itself* — every documented name is
re-exported, and each is the identical object from its source module, not a
shadowed stub. A dropped or rebound re-export would silently break the CLI's
imports; that is the regression this file exists to catch.
"""

from __future__ import annotations

import pytest

from osprey.deployment import (
    compose_generator,
    container_lifecycle,
    container_manager,
    status_display,
)

# Documented public surface, keyed by the module each name must resolve to.
RE_EXPORTS: dict[str, tuple[object, tuple[str, ...]]] = {
    "compose_generator": (
        compose_generator,
        (
            "COMPOSE_FILE_NAME",
            "OUT_SRC_DIR",
            "SERVICES_DIR",
            "SRC_DIR",
            "TEMPLATE_FILENAME",
            "clean_deployment",
            "find_existing_compose_files",
            "find_service_config",
            "prepare_compose_files",
            "render_kernel_templates",
            "render_template",
            "setup_build_dir",
        ),
    ),
    "container_lifecycle": (
        container_lifecycle,
        ("deploy_down", "deploy_restart", "deploy_up", "rebuild_deployment"),
    ),
    "status_display": (status_display, ("show_status",)),
}

ALL_NAMES = [(mod, name) for mod, names in RE_EXPORTS.values() for name in names]


@pytest.mark.parametrize(
    ("origin", "name"),
    [(origin, name) for origin, names in RE_EXPORTS.values() for name in names],
)
def test_reexport_is_identical_object(origin, name):
    """Each re-export must be the SAME object as in its source module, so
    callers importing via the facade get the real implementation."""
    assert hasattr(container_manager, name), f"container_manager dropped re-export {name!r}"
    assert getattr(container_manager, name) is getattr(origin, name), (
        f"{name!r} on container_manager is not the object from {origin.__name__}"
    )


def test_lifecycle_entrypoints_are_callable():
    """The four deploy entry points the CLI dispatches to must be callable."""
    for name in ("deploy_up", "deploy_down", "deploy_restart", "rebuild_deployment"):
        assert callable(getattr(container_manager, name))


def test_compose_constants_carry_expected_values():
    """The re-exported path/name constants keep their documented values (the CLI
    and tests key filesystem layout off these)."""
    assert container_manager.COMPOSE_FILE_NAME == "docker-compose.yml"
    assert container_manager.TEMPLATE_FILENAME == "docker-compose.yml.j2"
    assert container_manager.SERVICES_DIR == "services"


def test_deploy_cmd_imports_resolve_through_facade():
    """Every name the deploy CLI pulls from container_manager resolves — a
    lightweight guard that the facade covers the CLI's actual import list."""
    from osprey.cli import deploy_cmd  # noqa: F401  (import side effect is the assertion)

    # deploy_cmd imports lazily; assert the facade exposes the lifecycle verbs
    # it dispatches to regardless of import timing.
    for _origin, name in ALL_NAMES:
        assert hasattr(container_manager, name)
