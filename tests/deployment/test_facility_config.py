"""Tests for the facility-config `gitlab:` → `ci:` normalizer."""

import ast
import copy
import inspect
import re
import warnings
from pathlib import Path

import pytest
import yaml

from osprey.deployment.facility_config import (
    _reset_warn_state,
    normalize_facility_config,
)

GITLAB_BLOCK = {
    "host": "git.dls.example.org",
    "remote_name": "gitlab",
    "default_branch": "main",
    "project_id": 1234,
    "project_path": "physics/production/dls-profiles",
    "token_env_var": "DLS_GITLAB_TOKEN",
}

CI_BLOCK = {"provider": "gitlab", **GITLAB_BLOCK}


def _base_config(**overrides):
    config = {
        "facility": {"name": "Demo Light Source", "prefix": "dls"},
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
    }
    config.update(overrides)
    return config


@pytest.fixture(autouse=True)
def _reset_warn_flag():
    _reset_warn_state()
    yield
    _reset_warn_state()


class TestGitlabAliasing:
    def test_gitlab_block_aliased_to_ci(self):
        config = _base_config(gitlab=dict(GITLAB_BLOCK))

        normalized = normalize_facility_config(config)

        assert "gitlab" not in normalized
        assert normalized["ci"] == CI_BLOCK

    def test_ci_block_passes_through_unchanged(self):
        config = _base_config(ci=dict(CI_BLOCK))

        normalized = normalize_facility_config(config)

        assert normalized["ci"] == CI_BLOCK
        assert "gitlab" not in normalized

    def test_both_present_ci_wins(self):
        conflicting_gitlab = {**GITLAB_BLOCK, "token_env_var": "OTHER_TOKEN"}
        config = _base_config(gitlab=conflicting_gitlab, ci=dict(CI_BLOCK))

        normalized = normalize_facility_config(config)

        assert normalized["ci"] == CI_BLOCK
        assert "gitlab" not in normalized

    def test_both_present_emits_warning(self):
        config = _base_config(gitlab=dict(GITLAB_BLOCK), ci=dict(CI_BLOCK))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            normalize_facility_config(config)

        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_no_gitlab_no_ci_no_warning(self):
        config = _base_config()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            normalized = normalize_facility_config(config)

        assert "ci" not in normalized
        assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


class TestWarnOnce:
    def test_exactly_one_warning_per_process(self):
        config = _base_config(gitlab=dict(GITLAB_BLOCK))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            normalize_facility_config(config)
            normalize_facility_config(copy.deepcopy(config))
            normalize_facility_config(copy.deepcopy(config))

        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1

    def test_reset_allows_warning_again(self):
        config = _base_config(gitlab=dict(GITLAB_BLOCK))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            normalize_facility_config(config)
            _reset_warn_state()
            normalize_facility_config(copy.deepcopy(config))

        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 2


class TestRegistryTokenDefault:
    def test_registry_token_defaults_from_ci(self):
        config = _base_config(gitlab=dict(GITLAB_BLOCK))

        normalized = normalize_facility_config(config)

        assert normalized["registry"]["token_env_var"] == GITLAB_BLOCK["token_env_var"]

    def test_registry_token_not_overridden_when_set(self):
        config = _base_config(gitlab=dict(GITLAB_BLOCK))
        config["registry"]["token_env_var"] = "CUSTOM_REGISTRY_TOKEN"

        normalized = normalize_facility_config(config)

        assert normalized["registry"]["token_env_var"] == "CUSTOM_REGISTRY_TOKEN"

    def test_no_registry_block_does_not_error(self):
        config = {"gitlab": dict(GITLAB_BLOCK)}

        normalized = normalize_facility_config(config)

        assert normalized["ci"] == CI_BLOCK
        assert "registry" not in normalized


class TestNonMutating:
    def test_input_dict_unchanged(self):
        config = _base_config(gitlab=dict(GITLAB_BLOCK))
        original = copy.deepcopy(config)

        normalize_facility_config(config)

        assert config == original

    def test_returns_new_dict(self):
        config = _base_config(gitlab=dict(GITLAB_BLOCK))

        normalized = normalize_facility_config(config)

        assert normalized is not config
        assert normalized["registry"] is not config["registry"]


class TestEquivalence:
    def test_gitlab_shape_equals_ci_shape(self):
        gitlab_shape = _base_config(gitlab=dict(GITLAB_BLOCK))
        ci_shape = _base_config(ci=dict(CI_BLOCK))

        normalized_gitlab = normalize_facility_config(gitlab_shape)
        normalized_ci = normalize_facility_config(ci_shape)

        assert normalized_gitlab == normalized_ci


# =============================================================================
# Load-site wiring: normalize_facility_config() must run at every place the
# codebase reads a facility-config.yml (or a project config.yml carrying the
# same facility/registry/ci blocks forward), so no consumer ever sees a raw
# `gitlab:` block.
# =============================================================================

from osprey.cli import scaffold_cmd  # noqa: E402
from osprey.deployment import (  # noqa: E402
    compose_generator,
    container_lifecycle,
    status_display,
)
from osprey.deployment.web_terminals import lifecycle as wt_lifecycle  # noqa: E402
from osprey.deployment.web_terminals import seeding as wt_seeding  # noqa: E402

# Every known function that reads a facility config from disk (directly via
# ``yaml.safe_load`` or via ``ConfigBuilder(...).raw_config``) and must funnel
# the result through ``normalize_facility_config()``. Keep this list honest:
# add an entry whenever a new load site is wired.
_WIRED_LOAD_SITES = [
    (scaffold_cmd, "_load_facility_config"),
    (compose_generator, "prepare_compose_files"),
    (container_lifecycle, "deploy_down"),
    (status_display, "show_status"),
    (wt_lifecycle, "decommission_user"),
    (wt_lifecycle, "prune_users"),
    (wt_lifecycle, "nuke_stack"),
    (wt_seeding, "seed_web_terminals"),
]

# Modules scanned for *any* `.raw_config` read, so a future load site added to
# one of these modules without wiring the normalizer fails
# ``test_no_unwrapped_raw_config_reads`` below, not just a change to one of the
# functions named in ``_WIRED_LOAD_SITES`` above.
_SCANNED_MODULES = [
    compose_generator,
    container_lifecycle,
    status_display,
    wt_lifecycle,
    wt_seeding,
]

# Exact (stripped) source lines that legitimately read `.raw_config` WITHOUT
# calling normalize_facility_config() — each is a considered exception, not an
# oversight, and must justify itself here.
_ALLOWED_UNWRAPPED_LINES = {
    # container_lifecycle._resolve_persona_claude_cli_version reads a PERSONA
    # project's own config.yml (never the facility config being deployed) —
    # see that function's docstring. It only reads claude_code.cli_version,
    # so it never needs the gitlab/ci/registry normalization.
    "persona_config = ConfigBuilder(str(config_path)).raw_config",
}

_RAW_CONFIG_RE = re.compile(r"\.raw_config\b")

# Substrings that indicate a function reads a COMPLETE config off disk another
# way than `.raw_config` — a bare (no-path) `ConfigBuilder()` singleton read,
# or `.get_unexpanded_config()` (the unexpanded-placeholder sibling of
# `.raw_config`, used where secrets must not be flattened to disk). Any
# function containing one of these markers must also call
# normalize_facility_config() somewhere in its body, or be named in
# ``_ALLOWED_UNWRAPPED_FUNCTIONS`` below with a documented reason.
_FULL_CONFIG_READ_MARKERS = ("ConfigBuilder()", "get_unexpanded_config(")

# (module.__name__, function_name): reason. Empty today — every function that
# reads a full config via ConfigBuilder()/get_unexpanded_config() also
# normalizes it (compose_generator.setup_build_dir). Add an entry here only
# with the same rigor as _ALLOWED_UNWRAPPED_LINES above.
_ALLOWED_UNWRAPPED_FUNCTIONS: dict[tuple[str, str], str] = {}


def _module_functions(module):
    """Yield (name, unparsed source) for every function def in ``module``.

    Uses the AST rather than a line sweep because the bare-``ConfigBuilder()``
    read and its ``normalize_facility_config()`` wrap can legitimately land on
    different lines (see ``setup_build_dir``, which reads then normalizes a
    few lines later) — unlike the single-expression ``.raw_config`` sites,
    which the line-based sweep above handles.
    """
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, ast.unparse(node)


class TestLoadSiteWiring:
    """Statically enumerates facility-config load sites and guards their wiring."""

    @pytest.mark.parametrize(
        "module, func_name",
        _WIRED_LOAD_SITES,
        ids=[f"{m.__name__}.{f}" for m, f in _WIRED_LOAD_SITES],
    )
    def test_known_load_site_calls_normalizer(self, module, func_name):
        func = getattr(module, func_name)
        source = inspect.getsource(func)
        assert "normalize_facility_config(" in source, (
            f"{module.__name__}.{func_name} reads a facility config but never calls "
            "normalize_facility_config() — every facility-config load site must funnel "
            "through the normalizer."
        )

    @pytest.mark.parametrize("module", _SCANNED_MODULES, ids=lambda m: m.__name__)
    def test_no_unwrapped_raw_config_reads(self, module):
        """No `.raw_config` read in a deploy module bypasses the normalizer.

        Catches a *future* load site (not just the ones named in
        ``_WIRED_LOAD_SITES``) that reads ``ConfigBuilder(...).raw_config``
        without piping it through ``normalize_facility_config()`` on the same
        line — the convention every wired site here follows.
        """
        source = inspect.getsource(module)
        offenders = [
            stripped
            for line in source.splitlines()
            if _RAW_CONFIG_RE.search(stripped := line.strip())
            and "normalize_facility_config(" not in stripped
            and stripped not in _ALLOWED_UNWRAPPED_LINES
        ]
        assert not offenders, (
            f"{module.__name__} reads `.raw_config` without normalize_facility_config(): "
            f"{offenders}. Either wire it or add a documented exception to "
            "_ALLOWED_UNWRAPPED_LINES in this test."
        )

    @pytest.mark.parametrize("module", _SCANNED_MODULES, ids=lambda m: m.__name__)
    def test_no_unwrapped_full_config_reads(self, module):
        """No bare ``ConfigBuilder()`` / ``get_unexpanded_config()`` read escapes
        the normalizer either — the ``.raw_config`` sweep above is structurally
        blind to this accessor (e.g. ``compose_generator.setup_build_dir``,
        which stages a container's ``config.yml`` straight from
        ``ConfigBuilder().get_unexpanded_config()``).
        """
        offenders = []
        for func_name, func_src in _module_functions(module):
            if (module.__name__, func_name) in _ALLOWED_UNWRAPPED_FUNCTIONS:
                continue
            if any(marker in func_src for marker in _FULL_CONFIG_READ_MARKERS):
                if "normalize_facility_config(" not in func_src:
                    offenders.append(func_name)
        assert not offenders, (
            f"{module.__name__} has function(s) reading a full config via "
            f"ConfigBuilder()/get_unexpanded_config() without normalize_facility_config(): "
            f"{offenders}. Either wire it or add a documented exception to "
            "_ALLOWED_UNWRAPPED_FUNCTIONS in this test."
        )

    def test_scaffold_yaml_load_site_calls_normalizer(self):
        """The scaffold lint/render path parses facility-config.yml directly (no
        ConfigBuilder), so it can't be caught by the `.raw_config` sweep above."""
        source = inspect.getsource(scaffold_cmd._load_facility_config)
        assert "yaml.safe_load(f)" in source
        assert "normalize_facility_config(" in source


class TestLoadSiteWarnOnce:
    """Repeated loads across *different* load sites still emit one warning."""

    def test_repeated_loads_across_load_sites_emit_one_warning(self, tmp_path):
        config_path = tmp_path / "facility-config.yml"
        config_path.write_text(yaml.dump(_base_config(gitlab=dict(GITLAB_BLOCK))))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Site 1: the scaffold lint/render path (direct yaml.safe_load).
            scaffold_cmd._load_facility_config(str(config_path))
            # Site 2: a ConfigBuilder-based deploy-lifecycle path. web_terminals
            # is absent, so seed_web_terminals no-ops after loading + normalizing.
            wt_seeding.seed_web_terminals(str(config_path))

        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1


class TestLoadSiteBehaviorEquivalence:
    """A gitlab-shape config behaves identically to a ci-shape config through a
    real wired load site (not just through normalize_facility_config() directly)."""

    def test_scaffold_load_site_gitlab_equals_ci(self, tmp_path):
        gitlab_path = tmp_path / "gitlab-facility-config.yml"
        gitlab_path.write_text(yaml.dump(_base_config(gitlab=dict(GITLAB_BLOCK))))

        ci_path = tmp_path / "ci-facility-config.yml"
        ci_path.write_text(yaml.dump(_base_config(ci=dict(CI_BLOCK))))

        loaded_gitlab = scaffold_cmd._load_facility_config(str(gitlab_path))
        loaded_ci = scaffold_cmd._load_facility_config(str(ci_path))

        assert loaded_gitlab == loaded_ci
        assert loaded_gitlab["ci"] == CI_BLOCK
        assert "gitlab" not in loaded_gitlab

    def test_setup_build_dir_stages_normalized_config(self, tmp_path, monkeypatch):
        """A gitlab-shape project config.yml, staged into a container build dir
        by ``setup_build_dir``, comes out in the canonical `ci:` shape.

        Drives the real container-config propagation path: ``setup_build_dir``
        reads the CWD's ``config.yml`` via a bare ``ConfigBuilder()`` (not
        ``.raw_config``, which is why this needs its own test rather than
        reusing ``TestLoadSiteBehaviorEquivalence``'s scaffold-path fixture),
        flattens it, and writes the result as the service's staged
        ``config.yml`` — the file actually bind-mounted/copied into the
        container. That staged file must never carry the raw `gitlab:` block.
        """
        from osprey.deployment.compose_generator import setup_build_dir

        project_config_path = tmp_path / "config.yml"
        project_config_path.write_text(
            yaml.dump(_base_config(gitlab=dict(GITLAB_BLOCK), project_name="pep-fixture"))
        )

        service_dir = tmp_path / "services" / "worker"
        service_dir.mkdir(parents=True)
        (service_dir / "docker-compose.yml.j2").write_text(
            "services:\n  worker:\n    image: test\n"
        )

        monkeypatch.chdir(tmp_path)

        template_path = str(Path("services") / "worker" / "docker-compose.yml.j2")
        config = {"project_name": "pep-fixture", "build_dir": "./build"}
        container_cfg = {"copy_src": False, "render_kernel_templates": False}

        setup_build_dir(template_path, config, container_cfg)

        staged_config_path = tmp_path / "build" / "services" / "worker" / "config.yml"
        assert staged_config_path.is_file(), f"expected a staged config.yml at {staged_config_path}"
        staged_config = yaml.safe_load(staged_config_path.read_text())
        assert staged_config["ci"] == CI_BLOCK
        assert "gitlab" not in staged_config
