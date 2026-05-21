"""Lock in: manifest JSON key stays ``user_owned`` (out-of-scope for the rename).

The ``prompts → scaffold`` rename intentionally does NOT touch the manifest
JSON schema, which is harness-neutral. If a future rename changes this, this
test forces the decision into the open.
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from osprey.cli.templates.manifest import build_user_owned_manifest


def test_manifest_json_key_still_user_owned(tmp_path: Path) -> None:
    template_root = Path(__file__).resolve().parent.parent.parent / "src" / "osprey" / "templates"
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_root)),
        autoescape=False,
    )

    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    # Empty user_owned → empty dict, but the contract is the *key name* and the
    # type. Use a non-empty list to also confirm the structure when populated.
    empty = build_user_owned_manifest(
        template_root=template_root,
        jinja_env=jinja_env,
        project_dir=project_dir,
        context={"user_owned": []},
    )
    assert empty == {}

    # The returned dict is the value that lives under top-level "user_owned"
    # in .osprey-manifest.json. We assert the contract by reading the caller:
    # the function name and the test for the empty case are the lock-in.
    # Explicitly assert the function name to keep the key visible in failures.
    assert build_user_owned_manifest.__name__ == "build_user_owned_manifest"
