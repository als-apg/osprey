"""Tests for the ontology compiler's schema-loading stage.

Two contracts are pinned here.  The first is error funnelling: every way a
LinkML schema can fail to load — a misspelled slot, unparseable YAML, a file
that is not there — must surface as :class:`OntologyCompileError` naming the
schema file, because that error is what the CLI turns into a
:class:`click.ClickException` an operator reads.  ``linkml_runtime`` raises
three unrelated types for those three cases (:class:`TypeError`,
:class:`yaml.YAMLError`, :class:`OSError`), so the wrapping is load-bearing
rather than cosmetic.

The second is import isolation.  ``linkml_runtime`` imports ``rdflib``, which
is exactly what ``test_import_isolation.py`` keeps out of the facility-knowledge
runtime read path; this package is a sibling of ``ttl_generator/`` for that
reason.  A subprocess check here proves that importing the compiler package
does not load ``linkml_runtime`` at all — the guarantee that makes the sibling
arrangement safe rather than merely tidy.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from osprey.services.facility_knowledge.ontology_compiler.errors import OntologyCompileError
from osprey.services.facility_knowledge.ontology_compiler.loader import load_schema

#: Ceiling on every subprocess check.  Schema loading is local work; a hang
#: means something reached for the network, and that is a failure, not a wait.
SUBPROCESS_TIMEOUT_S = 120

#: A minimal but complete schema in the shape the compiler expects: narad_sem
#: prefix, ``linkml:types`` import, a root class and a child, and a
#: ``DeviceFamily`` enum.  Deliberately self-contained — the shipped
#: ``demo_ontology.yaml`` is authored by a different task, and these tests must
#: not wait on it.
GOOD_SCHEMA = textwrap.dedent(
    """\
    id: https://example.org/test_ontology
    name: test_ontology
    prefixes:
      narad_sem: https://narad.example.org/schema/shared_semantics/
      linkml: https://w3id.org/linkml/
    default_prefix: narad_sem
    imports:
      - linkml:types
    default_range: string
    classes:
      AcceleratorDevice:
        class_uri: narad_sem:AcceleratorDevice
        description: Root of the device vocabulary.
      Magnet:
        is_a: AcceleratorDevice
        class_uri: narad_sem:Magnet
        aliases:
          - magnet
          - bending magnet
    enums:
      DeviceFamily:
        permissible_values:
          MAGNET:
            meaning: narad_sem:Magnet
    """
)

#: The same schema with ``aliases:`` misspelled.  ``linkml_runtime`` passes
#: unknown slots straight to a generated dataclass ``__init__``, so this is a
#: :class:`TypeError` — the case a naive ``except ValueError`` would miss.
MISSPELLED_SLOT_SCHEMA = GOOD_SCHEMA.replace("    aliases:", "    aliasees:")

#: YAML that no parser can follow, to reach the :class:`yaml.YAMLError` branch.
MALFORMED_YAML_SCHEMA = "classes:\n  Foo:\n   - a\n  : : :\n\t bad\n"


def write_schema(tmp_path: Path, text: str, name: str = "schema.yaml") -> Path:
    """Write *text* to a schema file under *tmp_path* and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter and return the finished process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )


class TestLoadSchemaAccepts:
    """A well-formed schema loads, with its imports resolved."""

    def test_good_schema_loads(self, tmp_path: Path):
        """The authored classes come back through the view."""
        view = load_schema(write_schema(tmp_path, GOOD_SCHEMA))
        assert sorted(view.all_classes(imports=False)) == ["AcceleratorDevice", "Magnet"]
        assert view.get_class("Magnet").is_a == "AcceleratorDevice"
        assert list(view.get_class("Magnet").aliases) == ["magnet", "bending magnet"]

    def test_prefixes_expand(self, tmp_path: Path):
        """``narad_sem:`` resolves, which the payload stage depends on."""
        view = load_schema(write_schema(tmp_path, GOOD_SCHEMA))
        assert (
            view.expand_curie("narad_sem:Magnet")
            == "https://narad.example.org/schema/shared_semantics/Magnet"
        )

    def test_imports_closure_resolves_linkml_types(self, tmp_path: Path):
        """``imports: [linkml:types]`` is resolved, not merely recorded.

        The types schema ships inside ``linkml_runtime``; if the closure only
        listed the import without loading it, ``all_types()`` would be empty.
        """
        view = load_schema(write_schema(tmp_path, GOOD_SCHEMA))
        assert "linkml:types" in view.imports_closure()
        assert "string" in view.all_types()

    def test_linkml_types_import_resolves_offline(self, tmp_path: Path):
        """No network is touched while loading a schema that imports ``linkml:types``.

        The ``knowledge`` extra now carries network-capable dependencies, so
        this is checked rather than assumed: the child interpreter makes every
        socket constructor raise before ``linkml_runtime`` is imported.  A
        schema that reached out for ``https://w3id.org/linkml/types`` would
        fail here even on a connected machine.
        """
        source = write_schema(tmp_path, GOOD_SCHEMA)
        result = run_python(
            f"""
            import socket
            import ssl  # noqa: F401 - imported before the patch; it subclasses socket.socket

            class _NoNetworkSocket(socket.socket):
                def __init__(self, *args, **kwargs):
                    raise AssertionError("network access attempted during schema load")

            def _no_connection(*args, **kwargs):
                raise AssertionError("network access attempted during schema load")

            socket.socket = _NoNetworkSocket
            socket.create_connection = _no_connection

            try:
                socket.socket()
            except AssertionError:
                pass
            else:
                raise AssertionError("the network guard itself is not armed")

            from pathlib import Path

            from osprey.services.facility_knowledge.ontology_compiler.loader import load_schema

            view = load_schema(Path({str(source)!r}))
            assert "linkml:types" in view.imports_closure()
            assert "string" in view.all_types()
            print("OK")
            """
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout


class TestLoadSchemaRejects:
    """Every load failure becomes an ``OntologyCompileError`` naming the file."""

    def test_misspelled_slot(self, tmp_path: Path):
        """``aliasees:`` — a ``TypeError`` underneath — is wrapped and explained."""
        source = write_schema(tmp_path, MISSPELLED_SLOT_SCHEMA, name="misspelled.yaml")
        with pytest.raises(OntologyCompileError) as excinfo:
            load_schema(source)
        assert "misspelled.yaml" in str(excinfo.value)
        assert "aliasees" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, TypeError)

    def test_malformed_yaml(self, tmp_path: Path):
        """Unparseable YAML is wrapped, not raised as a bare ``YAMLError``."""
        source = write_schema(tmp_path, MALFORMED_YAML_SCHEMA, name="broken.yaml")
        with pytest.raises(OntologyCompileError) as excinfo:
            load_schema(source)
        assert "broken.yaml" in str(excinfo.value)

    def test_missing_file(self, tmp_path: Path):
        """A path that does not exist is an authoring error, not an ``OSError``."""
        source = tmp_path / "absent.yaml"
        with pytest.raises(OntologyCompileError) as excinfo:
            load_schema(source)
        assert "absent.yaml" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, OSError)

    def test_error_carries_the_source_path(self, tmp_path: Path):
        """The whole path is kept on the exception, not just the name."""
        source = write_schema(tmp_path, MALFORMED_YAML_SCHEMA, name="broken.yaml")
        with pytest.raises(OntologyCompileError) as excinfo:
            load_schema(source)
        assert excinfo.value.source == source
        assert excinfo.value.message

    def test_error_is_a_value_error(self, tmp_path: Path):
        """Matches ``OntologyMapError``, so one ``except`` covers both halves."""
        source = write_schema(tmp_path, MALFORMED_YAML_SCHEMA)
        with pytest.raises(ValueError):
            load_schema(source)


class TestPackageImportIsolation:
    """Importing the compiler package must not load the LinkML toolchain."""

    @pytest.mark.parametrize(
        "module",
        [
            "osprey.services.facility_knowledge.ontology_compiler",
            "osprey.services.facility_knowledge.ontology_compiler.errors",
            "osprey.services.facility_knowledge.ontology_compiler.loader",
        ],
    )
    def test_import_does_not_pull_linkml_runtime(self, module: str):
        """``linkml_runtime`` (and with it ``rdflib``) stays out of ``sys.modules``.

        Run in a fresh interpreter so a ``linkml_runtime`` already imported by
        another test in this session cannot make the check pass vacuously.
        """
        result = run_python(
            f"""
            import {module}  # noqa: F401
            import sys

            leaked = [m for m in sys.modules if m == "linkml_runtime" or m.startswith("linkml_runtime.")]
            assert not leaked, "linkml_runtime leaked into sys.modules: " + repr(leaked)
            leaked_rdflib = [m for m in sys.modules if m == "rdflib" or m.startswith("rdflib.")]
            assert not leaked_rdflib, "rdflib leaked into sys.modules: " + repr(leaked_rdflib)
            print("OK")
            """
        )
        assert result.returncode == 0, result.stderr

    def test_isolation_check_would_fire(self):
        """Negative control: the detector catches a real ``linkml_runtime`` import."""
        result = run_python(
            """
            import linkml_runtime  # noqa: F401
            import sys

            leaked = [m for m in sys.modules if m == "linkml_runtime" or m.startswith("linkml_runtime.")]
            assert not leaked, "linkml_runtime leaked into sys.modules: " + repr(leaked)
            print("OK")
            """
        )
        assert result.returncode != 0, "detector passed on a module that DOES import linkml_runtime"


class TestLazyExports:
    """The package re-exports through ``__getattr__``, so absent stages cost nothing."""

    def test_error_type_is_reachable_from_the_package(self):
        """``OntologyCompileError`` resolves to the class ``errors`` defines."""
        import osprey.services.facility_knowledge.ontology_compiler as pkg

        assert pkg.OntologyCompileError is OntologyCompileError

    def test_loader_entry_point_is_reachable_from_the_package(self):
        """``load_schema`` resolves without importing ``linkml_runtime``."""
        import osprey.services.facility_knowledge.ontology_compiler as pkg

        assert pkg.load_schema is load_schema

    def test_unknown_attribute_raises_attribute_error(self):
        """A typo is an ``AttributeError``, not an ``ImportError`` from nowhere."""
        import osprey.services.facility_knowledge.ontology_compiler as pkg

        with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
            pkg.nonexistent

    def test_public_names_are_declared(self):
        """``__all__`` names the compiler's documented surface."""
        import osprey.services.facility_knowledge.ontology_compiler as pkg

        assert set(pkg.__all__) == {
            "OntologyCompileError",
            "CompiledOntology",
            "compile_schema",
            "render_json",
            "check_artifact",
            "GENERATED_HEADER",
        }
