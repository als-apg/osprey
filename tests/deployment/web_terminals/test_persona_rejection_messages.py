"""What a rejected persona ``build_profile`` tells the operator to do next.

The shapes themselves are rejected elsewhere (``test_persona_images.py`` for
the deploy path, ``test_lint.py`` for the gate). What these tests pin is the
remedy, on both paths at once: the operator most likely to hit either
rejection is one whose variant build predates the persona-delta layout, so
"set this value to ``personas/<name>.yml``" is advice about a file they do not
have. Both messages must therefore also name ``/osprey-build-interview``, the
thing that converts an old variant into that file — and they must keep saying
the same thing as each other, since a config can be rejected by whichever path
the operator happens to reach first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.deployment.web_terminals import persona_images
from osprey.deployment.web_terminals.lint import lint_web_terminals

#: A pre-delta value: the bundled preset name a facility used to write here.
_LEGACY_VALUE = "control-assistant"

_INTERVIEW = "/osprey-build-interview"


def _profile_root(tmp_path: Path) -> Path:
    root = tmp_path / "facility-profile"
    (root / "personas").mkdir(parents=True)
    (root / "profile.yml").write_text("name: facility-profile\n")
    return root


def _deploy_rejection(tmp_path: Path, build_profile: str) -> str:
    """The deploy path's message for a build_profile it refuses."""
    with pytest.raises(ValueError) as excinfo:
        persona_images._resolve_persona_profile(build_profile, "ops", _profile_root(tmp_path))
    return str(excinfo.value)


def _lint_rejection(tmp_path: Path, build_profile: str) -> str:
    """The gate's message for the same build_profile."""
    findings = lint_web_terminals(
        {
            "modules": {
                "web_terminals": {
                    "enabled": True,
                    "image_source": "local",
                    "default_persona": "ops",
                    "personas": {
                        "ops": {
                            "project": "ops",
                            "project_path": str(tmp_path / "ops"),
                            "build_profile": build_profile,
                        }
                    },
                    "users": [{"name": "thellert", "index": 0}],
                }
            }
        }
    )
    shape = [f for f in findings if f.code == "web_terminals.persona_build_profile_not_a_delta"]
    assert shape, f"{build_profile!r} drew no shape finding: {[f.code for f in findings]}"
    return shape[0].message


def test_deploy_rejection_names_the_build_interview(tmp_path: Path):
    assert _INTERVIEW in _deploy_rejection(tmp_path, _LEGACY_VALUE)


def test_lint_rejection_names_the_build_interview(tmp_path: Path):
    assert _INTERVIEW in _lint_rejection(tmp_path, _LEGACY_VALUE)


def test_missing_delta_file_also_names_the_build_interview(tmp_path: Path):
    """A well-shaped value pointing at nothing is the *likeliest* pre-delta
    symptom: the operator took the advice, wrote the path, and has no file to
    put there. That message shares the same remedy sentence, so it inherits
    the pointer rather than dead-ending."""
    message = _deploy_rejection(tmp_path, "personas/ops.yml")

    assert "no file exists at" in message
    assert _INTERVIEW in message


@pytest.mark.parametrize(
    "build_profile",
    [
        _LEGACY_VALUE,  # bundled preset name
        "/abs/personas/ops.yml",  # absolute path
        "profiles/ops.yml",  # the registry-mode lone-YAML spelling
    ],
)
def test_both_paths_recommend_the_same_delta_and_the_interview(tmp_path: Path, build_profile: str):
    """One predicate decides the shape, but the two callers write their own
    remedies. If they drift, an operator who lints clean and then fails the
    deploy gets contradictory advice about which file to create."""
    deploy_message = _deploy_rejection(tmp_path, build_profile)
    lint_message = _lint_rejection(tmp_path, build_profile)

    for message in (deploy_message, lint_message):
        assert "personas/ops.yml" in message
        assert _INTERVIEW in message
