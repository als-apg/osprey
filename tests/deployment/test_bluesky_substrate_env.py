"""Deploy-time provisioning for the bluesky plan stack.

Covers the key material ``osprey up`` has to put in place before ``compose up``
mounts it, both in ``container_lifecycle``:

* ``_ensure_bluesky_control_plane_keys`` — the RE manager's control-socket
  CURVE keypair, which must *pair* and therefore cannot ride on the
  independent per-variable recipes in ``_ensure_service_tokens``.
* ``_ensure_bluesky_document_plane_certs`` — the document plane's CURVE
  certificates, generated unconditionally so a browse-only deploy that later
  flips its connector does not find the plane silently broken.

The plan devices are no longer among them: the worker reads an authored
devices file staged into the build context at render time
(``_stage_bluesky_devices`` in ``compose_generator``), so nothing is derived
into ``.env`` for them any more. What survives here is the template side of
that contract — ``TestComposeTemplateContract`` pins the container path and
mount source the staging step has to keep feeding.

The compose template (``templates/services/bluesky/docker-compose.yml.j2``) is
the binding contract for the paths and variable names asserted here.
"""

import os
import re
import stat
from pathlib import Path

import pytest

from osprey.deployment.container_lifecycle import (
    _bluesky_curve_paths,
    _ensure_bluesky_control_plane_keys,
    _ensure_bluesky_document_plane_certs,
)

BLUESKY_STACK = ["bluesky", "virtual_accelerator"]


@pytest.fixture
def env_path(tmp_path, monkeypatch):
    """A project ``.env`` in an isolated project dir, with a clean process env.

    Every variable these functions read is cleared from ``os.environ`` so a
    developer's own exported values cannot make a test pass or fail.
    """
    for var in (
        "BLUESKY_QSERVER_ZMQ_PRIVATE_KEY",
        "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path / ".env"


def _dotenv(env_path):
    """Parse the project ``.env``, or an empty mapping when it does not exist."""
    from osprey.utils.dotenv import parse_dotenv_file

    return parse_dotenv_file(env_path) if env_path.is_file() else {}


def _curve_keypair():
    """A CURVE keypair whose halves are both free of ``$``.

    Not cosmetic. ``_ensure_bluesky_control_plane_keys`` now REFUSES any
    control-plane key containing ``$``, because docker compose would expand it
    out of the project ``.env`` (see the interpolation section at the bottom of
    this file). A bare ``zmq.curve_keypair()`` therefore produces an unusable
    operator key roughly 62% of the time, and every test below that supplies a
    keypair to exercise some OTHER property — never-rotate, pair verification,
    derived public halves — would fail intermittently for a reason unrelated to
    what it is testing.

    Drawing usable keys here keeps those tests about their own subject. The
    refusal itself is pinned deliberately, and only, in
    ``TestControlPlaneKeysSurviveComposeInterpolation``.
    """
    import zmq

    while True:
        public, secret = zmq.curve_keypair()
        if b"$" not in public and b"$" not in secret:
            return public, secret


class TestControlPlaneKeys:
    """The RE manager's control-socket keypair, minted into the project .env."""

    def test_mints_a_matching_pair(self, env_path):
        import zmq

        _ensure_bluesky_control_plane_keys({"deployed_services": ["bluesky"]}, env_path=env_path)

        written = _dotenv(env_path)
        private = written["BLUESKY_QSERVER_ZMQ_PRIVATE_KEY"]
        public = written["BLUESKY_QSERVER_ZMQ_PUBLIC_KEY"]
        # The whole reason this cannot ride on _ensure_service_tokens: two
        # independently random keys are exactly a broken CURVE handshake.
        assert zmq.curve_public(private.encode()).decode() == public
        assert len(private) == 40  # z85-encoded 32-byte key
        assert private != public

    def test_env_file_is_owner_only(self, env_path):
        _ensure_bluesky_control_plane_keys({"deployed_services": ["bluesky"]}, env_path=env_path)

        assert stat.S_IMODE(env_path.stat().st_mode) == 0o600

    def test_no_bluesky_service_is_a_no_op(self, env_path):
        _ensure_bluesky_control_plane_keys({"deployed_services": ["postgresql"]}, env_path=env_path)

        assert not env_path.exists()

    def test_existing_pair_is_never_rotated(self, env_path):
        """A REAL operator-supplied pair survives a redeploy untouched.

        Uses a genuine keypair, not two placeholder strings: the pairing check
        would reject a mismatched pair, so placeholders would test the refusal
        path while claiming to test the never-rotate one.
        """

        public, secret = _curve_keypair()
        env_path.write_text(
            f"BLUESKY_QSERVER_ZMQ_PRIVATE_KEY={secret.decode()}\n"
            f"BLUESKY_QSERVER_ZMQ_PUBLIC_KEY={public.decode()}\n",
            encoding="utf-8",
        )
        before = env_path.read_text(encoding="utf-8")

        _ensure_bluesky_control_plane_keys({"deployed_services": ["bluesky"]}, env_path=env_path)

        assert env_path.read_text(encoding="utf-8") == before

    def test_mismatched_operator_pair_is_refused(self, env_path):
        """Two well-formed keys that do not belong together satisfy every compose guard.

        Left unchecked this starts a stack whose every queue call dies as an
        unexplained control-socket timeout, so it has to be caught here.
        """

        _, secret = _curve_keypair()
        other_public, _ = _curve_keypair()
        env_path.write_text(
            f"BLUESKY_QSERVER_ZMQ_PRIVATE_KEY={secret.decode()}\n"
            f"BLUESKY_QSERVER_ZMQ_PUBLIC_KEY={other_public.decode()}\n",
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError) as excinfo:
            _ensure_bluesky_control_plane_keys(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

        message = str(excinfo.value)
        assert "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY" in message
        assert "BLUESKY_QSERVER_ZMQ_PRIVATE_KEY" in message
        # Names the cause, never echoes key material.
        assert secret.decode() not in message
        assert other_public.decode() not in message

    def test_matched_operator_pair_passes_the_check(self, env_path):
        """The pairing check must not reject a pair that genuinely matches."""

        public, secret = _curve_keypair()
        env_path.write_text(
            f"BLUESKY_QSERVER_ZMQ_PRIVATE_KEY={secret.decode()}\n"
            f"BLUESKY_QSERVER_ZMQ_PUBLIC_KEY={public.decode()}\n",
            encoding="utf-8",
        )

        _ensure_bluesky_control_plane_keys({"deployed_services": ["bluesky"]}, env_path=env_path)

    def test_malformed_private_key_in_a_pair_warns_rather_than_raising(self, env_path, caplog):
        """Nothing can be verified against a key that will not parse."""

        public, _ = _curve_keypair()
        env_path.write_text(
            "BLUESKY_QSERVER_ZMQ_PRIVATE_KEY=not-a-curve-key\n"
            f"BLUESKY_QSERVER_ZMQ_PUBLIC_KEY={public.decode()}\n",
            encoding="utf-8",
        )

        with caplog.at_level("WARNING"):
            _ensure_bluesky_control_plane_keys(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

        assert "does not parse" in caplog.text

    def test_idempotent_across_redeploys(self, env_path):
        config = {"deployed_services": ["bluesky"]}

        _ensure_bluesky_control_plane_keys(config, env_path=env_path)
        first = _dotenv(env_path)
        _ensure_bluesky_control_plane_keys(config, env_path=env_path)

        assert _dotenv(env_path) == first

    def test_operator_private_key_gets_a_derived_public_half(self, env_path):
        """A hand-supplied manager key is kept, and its public half derived."""

        public, secret = _curve_keypair()
        env_path.write_text(
            f"BLUESKY_QSERVER_ZMQ_PRIVATE_KEY={secret.decode()}\n", encoding="utf-8"
        )

        _ensure_bluesky_control_plane_keys({"deployed_services": ["bluesky"]}, env_path=env_path)

        written = _dotenv(env_path)
        assert written["BLUESKY_QSERVER_ZMQ_PRIVATE_KEY"] == secret.decode()
        assert written["BLUESKY_QSERVER_ZMQ_PUBLIC_KEY"] == public.decode()

    def test_shell_only_private_key_leaves_no_half_state_on_disk(self, env_path, monkeypatch):
        """A private half that lives only in this shell must not persist its public half.

        Writing it would leave a public key on disk with no private one beside
        it, so the next deploy from a clean shell would hit the orphan branch
        and abort at the compose guard. Export it for this deploy and re-derive
        it next time instead.
        """

        public, secret = _curve_keypair()
        monkeypatch.setenv("BLUESKY_QSERVER_ZMQ_PRIVATE_KEY", secret.decode())
        # The mint exports the derived public half straight into os.environ;
        # recording its absence here is what makes monkeypatch take it back
        # out at teardown instead of leaking it to every later test in this
        # worker as an orphan public key.
        monkeypatch.delenv("BLUESKY_QSERVER_ZMQ_PUBLIC_KEY", raising=False)

        _ensure_bluesky_control_plane_keys({"deployed_services": ["bluesky"]}, env_path=env_path)

        # Nothing on disk — neither half.
        assert not env_path.exists() or "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY" not in _dotenv(env_path)
        # But this deploy's compose invocation still sees it: deploy_up copies
        # os.environ AFTER this runs, and the template guards the public key.
        assert os.environ["BLUESKY_QSERVER_ZMQ_PUBLIC_KEY"] == public.decode()

    def test_shell_only_private_key_is_rederived_each_deploy(self, env_path, monkeypatch):
        """Re-deriving is what makes the shell-only path self-healing."""

        public, secret = _curve_keypair()
        monkeypatch.setenv("BLUESKY_QSERVER_ZMQ_PRIVATE_KEY", secret.decode())
        # Recorded as absent BEFORE the first mint exports it, so teardown
        # removes the public half rather than restoring the exported value.
        monkeypatch.delenv("BLUESKY_QSERVER_ZMQ_PUBLIC_KEY", raising=False)
        config = {"deployed_services": ["bluesky"]}

        _ensure_bluesky_control_plane_keys(config, env_path=env_path)
        monkeypatch.delenv("BLUESKY_QSERVER_ZMQ_PUBLIC_KEY", raising=False)
        _ensure_bluesky_control_plane_keys(config, env_path=env_path)

        assert os.environ["BLUESKY_QSERVER_ZMQ_PUBLIC_KEY"] == public.decode()

    def test_dotenv_private_key_does_persist_its_derived_public_half(self, env_path):
        """The disk-backed case is the opposite: both halves end up on disk together."""

        public, secret = _curve_keypair()
        env_path.write_text(
            f"BLUESKY_QSERVER_ZMQ_PRIVATE_KEY={secret.decode()}\n", encoding="utf-8"
        )

        _ensure_bluesky_control_plane_keys({"deployed_services": ["bluesky"]}, env_path=env_path)

        assert _dotenv(env_path)["BLUESKY_QSERVER_ZMQ_PUBLIC_KEY"] == public.decode()

    def test_public_key_without_private_half_refuses_to_fabricate(self, env_path, caplog):
        """A secret key is not recoverable from a public one — warn, never mint."""
        env_path.write_text("BLUESKY_QSERVER_ZMQ_PUBLIC_KEY=orphan\n", encoding="utf-8")

        with caplog.at_level("WARNING"):
            _ensure_bluesky_control_plane_keys(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

        assert "BLUESKY_QSERVER_ZMQ_PRIVATE_KEY" not in _dotenv(env_path)
        # Minting a fresh private key here would orphan the operator's public
        # one; the warning has to say the deploy stops rather than degrades.
        assert "orphan" in caplog.text
        assert "fail-closed" in caplog.text

    def test_empty_private_key_is_refused_not_honoured(self, env_path, monkeypatch):
        """Plaintext is not a supported mode — empty is unsatisfiable, not a choice.

        The compose template guards this variable with `:?`, which rejects
        empty exactly as it rejects unset, so an empty value cannot produce a
        running stack. Refusing here turns a confusing abort deep inside
        compose into an actionable message.
        """
        monkeypatch.setenv("BLUESKY_QSERVER_ZMQ_PRIVATE_KEY", "")

        with pytest.raises(RuntimeError, match="plaintext"):
            _ensure_bluesky_control_plane_keys(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

    def test_empty_private_key_in_dotenv_is_refused_too(self, env_path):
        """Same ruling whichever source the empty value came from."""
        env_path.write_text("BLUESKY_QSERVER_ZMQ_PRIVATE_KEY=\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="plaintext"):
            _ensure_bluesky_control_plane_keys(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

    def test_empty_public_key_is_refused_too(self, env_path, monkeypatch):
        """Symmetric: an empty public half cannot authenticate to a keyed manager."""
        monkeypatch.setenv("BLUESKY_QSERVER_ZMQ_PUBLIC_KEY", "")

        with pytest.raises(RuntimeError, match="plaintext"):
            _ensure_bluesky_control_plane_keys(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

    def test_refusal_names_both_ways_out(self, env_path, monkeypatch):
        """The error is actionable: unset it, or supply your own private half."""
        monkeypatch.setenv("BLUESKY_QSERVER_ZMQ_PRIVATE_KEY", "   ")

        with pytest.raises(RuntimeError) as excinfo:
            _ensure_bluesky_control_plane_keys(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

        assert "unset" in str(excinfo.value)
        assert "private half" in str(excinfo.value)

    def test_malformed_private_key_never_raises_into_the_deploy(self, env_path, caplog):
        """A non-empty but invalid key is a warning: the `:?` guard stops the deploy."""
        env_path.write_text("BLUESKY_QSERVER_ZMQ_PRIVATE_KEY=not-a-curve-key\n", encoding="utf-8")

        with caplog.at_level("WARNING"):
            _ensure_bluesky_control_plane_keys(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

        assert "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY" not in _dotenv(env_path)
        assert "plaintext" in caplog.text

    def test_minted_on_a_mock_deploy_too(self, env_path):
        """Not gated on the connector: a later flip must not find the pair missing."""
        _ensure_bluesky_control_plane_keys(
            {"deployed_services": ["bluesky"], "control_system": {"type": "mock"}},
            env_path=env_path,
        )

        assert "BLUESKY_QSERVER_ZMQ_PRIVATE_KEY" in _dotenv(env_path)

    def test_neither_half_is_ever_logged(self, env_path, caplog):
        """Both halves are bearer material — logs carry variable names only.

        Upstream ``bluesky-queueserver`` clients authenticate with a fixed,
        publicly-known client keypair, so the manager's *public* key alone
        reaches its control socket. It is a secret here, not a publishable one.
        """
        with caplog.at_level("DEBUG"):
            _ensure_bluesky_control_plane_keys(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

        written = _dotenv(env_path)
        assert written["BLUESKY_QSERVER_ZMQ_PRIVATE_KEY"] not in caplog.text
        assert written["BLUESKY_QSERVER_ZMQ_PUBLIC_KEY"] not in caplog.text
        # The names are logged, so this is not passing by logging nothing at all.
        assert "BLUESKY_QSERVER_ZMQ_PRIVATE_KEY" in caplog.text

    def test_derived_public_half_is_not_logged_either(self, env_path, caplog):
        """The derive-from-operator-key branch has its own log line — same rule."""

        public, secret = _curve_keypair()
        env_path.write_text(
            f"BLUESKY_QSERVER_ZMQ_PRIVATE_KEY={secret.decode()}\n", encoding="utf-8"
        )

        with caplog.at_level("DEBUG"):
            _ensure_bluesky_control_plane_keys(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

        # Positive anchor first: this branch DOES log, naming the variable, so
        # the negative assertions below cannot pass on an empty caplog.
        assert "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY" in caplog.text
        assert public.decode() not in caplog.text
        assert secret.decode() not in caplog.text


class TestDocumentPlaneCerts:
    """The document plane's CURVE certificates, generated into the project tree."""

    def test_generates_the_full_split_layout(self, env_path):
        """Each side holds its own secret and the other's public key — never both secrets."""
        _ensure_bluesky_document_plane_certs({"deployed_services": ["bluesky"]}, env_path=env_path)

        paths = _bluesky_curve_paths(env_path.parent)
        for role in ("proxy_secret", "publisher_secret", "proxy_public", "publisher_public"):
            assert paths[role].is_file(), role

        # The mount source for the bridge must not carry the publisher's
        # secret, and vice versa — that split is the point of the layout.
        bridge_files = {p.name for p in paths["bridge"].rglob("*") if p.is_file()}
        queueserver_files = {p.name for p in paths["queueserver"].rglob("*") if p.is_file()}
        assert bridge_files == {"proxy.key_secret", "publisher.key"}
        assert queueserver_files == {"publisher.key_secret", "proxy.key"}

    def test_paths_match_the_compose_template(self, env_path):
        """The in-container paths the compose template hardcodes are these files.

        The template sets ``BLUESKY_ZMQ_CURVE_SECRET_KEY=/app/curve/proxy.key_secret``
        and ``BLUESKY_ZMQ_CURVE_CLIENT_PUBLIC_KEYS=/app/curve/clients`` on the
        bridge (mount source ``data/.runtime/bluesky_curve/bridge``), and
        ``/app/curve/publisher.key_secret`` + ``/app/curve/proxy.key`` on the
        queueserver (mount source ``data/.runtime/bluesky_curve/queueserver``).

        ``.runtime/`` is the reserved runtime-output subpath the build-drift
        fold skips (#716); the spelling is pinned literally because the writer,
        the fold, and the compose template must agree on these exact bytes.
        """
        paths = _bluesky_curve_paths(env_path.parent)
        rel = {k: v.relative_to(env_path.parent).as_posix() for k, v in paths.items()}

        assert rel["bridge"] == "data/.runtime/bluesky_curve/bridge"
        assert rel["queueserver"] == "data/.runtime/bluesky_curve/queueserver"
        assert rel["proxy_secret"] == "data/.runtime/bluesky_curve/bridge/proxy.key_secret"
        assert rel["publisher_public"] == "data/.runtime/bluesky_curve/bridge/clients/publisher.key"
        assert (
            rel["publisher_secret"]
            == "data/.runtime/bluesky_curve/queueserver/publisher.key_secret"
        )
        assert rel["proxy_public"] == "data/.runtime/bluesky_curve/queueserver/proxy.key"

    def test_certificates_are_a_usable_pair(self, env_path):
        """Loading each secret yields the public half its peer was given."""
        import zmq.auth

        _ensure_bluesky_document_plane_certs({"deployed_services": ["bluesky"]}, env_path=env_path)
        paths = _bluesky_curve_paths(env_path.parent)

        proxy_public, proxy_secret = zmq.auth.load_certificate(paths["proxy_secret"])
        assert proxy_secret is not None
        assert zmq.auth.load_certificate(paths["proxy_public"])[0] == proxy_public

        pub_public, pub_secret = zmq.auth.load_certificate(paths["publisher_secret"])
        assert pub_secret is not None
        assert zmq.auth.load_certificate(paths["publisher_public"])[0] == pub_public
        # Two independent keypairs, not one reused on both ends.
        assert proxy_public != pub_public

    def test_secret_certificates_are_owner_only(self, env_path):
        _ensure_bluesky_document_plane_certs({"deployed_services": ["bluesky"]}, env_path=env_path)
        paths = _bluesky_curve_paths(env_path.parent)

        assert stat.S_IMODE(paths["proxy_secret"].stat().st_mode) == 0o600
        assert stat.S_IMODE(paths["publisher_secret"].stat().st_mode) == 0o600

    def test_generated_for_a_mock_deploy_too(self, env_path):
        """Unconditional by design: a later connector flip must not find them missing."""
        _ensure_bluesky_document_plane_certs(
            {"deployed_services": ["bluesky"], "control_system": {"type": "mock"}},
            env_path=env_path,
        )

        assert _bluesky_curve_paths(env_path.parent)["proxy_secret"].is_file()

    def test_no_bluesky_service_is_a_no_op(self, env_path):
        _ensure_bluesky_document_plane_certs(
            {"deployed_services": ["postgresql"]}, env_path=env_path
        )

        assert not (env_path.parent / "data" / ".runtime" / "bluesky_curve").exists()

    def test_complete_set_is_never_rotated(self, env_path):
        """Redeploying must not pull keys out from under a running stack."""
        config = {"deployed_services": ["bluesky"]}
        _ensure_bluesky_document_plane_certs(config, env_path=env_path)
        paths = _bluesky_curve_paths(env_path.parent)
        before = {k: paths[k].read_bytes() for k in ("proxy_secret", "publisher_secret")}

        _ensure_bluesky_document_plane_certs(config, env_path=env_path)

        assert {k: paths[k].read_bytes() for k in before} == before

    def test_partial_set_is_regenerated_whole(self, env_path, caplog):
        """A secret half cannot be rebuilt from a public one, so repair means replace."""
        config = {"deployed_services": ["bluesky"]}
        _ensure_bluesky_document_plane_certs(config, env_path=env_path)
        paths = _bluesky_curve_paths(env_path.parent)
        stale_public = paths["proxy_public"].read_bytes()
        paths["proxy_secret"].unlink()

        with caplog.at_level("WARNING"):
            _ensure_bluesky_document_plane_certs(config, env_path=env_path)

        assert "incomplete" in caplog.text
        assert paths["proxy_secret"].is_file()
        # The surviving public half was replaced too — otherwise the pair the
        # publisher pins would no longer match the proxy's new secret.
        assert paths["proxy_public"].read_bytes() != stale_public

    def test_generation_failure_never_raises_into_the_deploy(self, env_path, caplog, monkeypatch):
        import zmq.auth

        def _boom(*args, **kwargs):
            raise OSError("no entropy today")

        monkeypatch.setattr(zmq.auth, "create_certificates", _boom)

        with caplog.at_level("WARNING"):
            _ensure_bluesky_document_plane_certs(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

        assert "no live rows" in caplog.text

    def test_no_scratch_directory_is_left_behind(self, env_path):
        """Certificates are staged in a temp dir; nothing of it survives."""
        _ensure_bluesky_document_plane_certs({"deployed_services": ["bluesky"]}, env_path=env_path)

        base = env_path.parent / "data" / ".runtime" / "bluesky_curve"
        assert {p.name for p in base.iterdir()} == {"bridge", "queueserver"}

    def test_no_key_material_is_logged(self, env_path, caplog):
        """Certificate *paths* are logged; the keys inside them never are."""
        import zmq.auth

        with caplog.at_level("DEBUG"):
            _ensure_bluesky_document_plane_certs(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

        paths = _bluesky_curve_paths(env_path.parent)
        # Anchor: the directory IS logged, so an empty caplog cannot make the
        # negative assertions below pass vacuously.
        assert str(paths["bridge"].parent) in caplog.text
        for role in ("proxy_secret", "publisher_secret"):
            public, secret = zmq.auth.load_certificate(paths[role])
            assert secret is not None
            assert secret.decode() not in caplog.text
            assert public.decode() not in caplog.text


class TestLegacyCurveDirMigration:
    """Certificates minted at the pre-#716 path relocate into ``data/.runtime/``.

    Releases before the reserved runtime-output subpath wrote each lane's set
    to ``data/<lane>_curve/`` — inside the folded build input, which is the bug.
    The writer relocates that material on the next ``up`` rather than leaving
    readers to check two paths: a move preserves the keys (rotation would break
    a running stack's pinned pairs), and afterwards exactly one spelling exists.
    """

    _LEGACY_LAYOUT = (
        ("bridge", "proxy.key_secret"),
        ("bridge", "clients", "publisher.key"),
        ("queueserver", "publisher.key_secret"),
        ("queueserver", "proxy.key"),
    )

    @classmethod
    def _write_legacy_set(cls, project_dir, lane_key="bluesky", parts_subset=None):
        legacy = project_dir / "data" / f"{lane_key}_curve"
        written = {}
        for parts in parts_subset or cls._LEGACY_LAYOUT:
            path = legacy.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            body = f"legacy {'/'.join(parts)}\n".encode()
            path.write_bytes(body)
            written["/".join(parts)] = body
        return legacy, written

    def test_complete_legacy_set_is_relocated_not_rotated(self, env_path):
        """The exact key bytes survive the move; nothing is re-minted."""
        legacy, written = self._write_legacy_set(env_path.parent)

        _ensure_bluesky_document_plane_certs({"deployed_services": ["bluesky"]}, env_path=env_path)

        base = env_path.parent / "data" / ".runtime" / "bluesky_curve"
        for rel, body in written.items():
            assert base.joinpath(*rel.split("/")).read_bytes() == body, rel
        assert not legacy.exists()

    def test_partial_legacy_set_is_relocated_then_repaired(self, env_path, caplog):
        """Relocation and the existing incomplete-set repair compose."""
        legacy, _ = self._write_legacy_set(
            env_path.parent, parts_subset=(("bridge", "proxy.key_secret"),)
        )

        with caplog.at_level("WARNING"):
            _ensure_bluesky_document_plane_certs(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

        assert not legacy.exists()
        paths = _bluesky_curve_paths(env_path.parent)
        for role in ("proxy_secret", "publisher_secret", "proxy_public", "publisher_public"):
            assert paths[role].is_file(), role
        assert "incomplete" in caplog.text

    def test_fresh_mint_is_untouched_by_migration(self, env_path):
        """No legacy directory, no relocation: the normal mint path still runs."""
        _ensure_bluesky_document_plane_certs({"deployed_services": ["bluesky"]}, env_path=env_path)

        assert _bluesky_curve_paths(env_path.parent)["proxy_secret"].is_file()
        assert not (env_path.parent / "data" / "bluesky_curve").exists()

    def test_runtime_dir_ignores_itself_for_legacy_repos(self, env_path):
        """A repo scaffolded before #716 has no ``/data/.runtime/`` ignore entry.

        Its gitignore covers only the legacy path, so after relocation the key
        material would show up as untracked — one ``git add -A`` away from
        committing secrets. The writer therefore drops a self-ignoring
        ``.gitignore`` inside ``data/.runtime/``, which needs no edit to any
        file the operator owns.
        """
        import subprocess

        repo = env_path.parent
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        # The pre-#716 scaffold ignored only the legacy spelling.
        (repo / ".gitignore").write_text("/data/bluesky_curve/\n", encoding="utf-8")
        self._write_legacy_set(repo)

        _ensure_bluesky_document_plane_certs({"deployed_services": ["bluesky"]}, env_path=env_path)

        ignored = subprocess.run(
            [
                "git",
                "check-ignore",
                "-q",
                "--no-index",
                "data/.runtime/bluesky_curve/bridge/proxy.key_secret",
            ],
            cwd=repo,
            capture_output=True,
        )
        assert ignored.returncode == 0, "relocated key material must stay out of version control"


class TestComposeTemplateContract:
    """What the compose template consumes is what this module must provide.

    A rename on either side is a hard ``osprey up`` abort rather than a soft
    failure — the template guards both halves with ``:?`` — so the two names
    are pinned here against the template's own output.

    Against the OUTPUT, not the source: the template renders one stack per
    Bluesky plan lane, so the variable names and mount sources below are
    per-lane expressions in the source and only become these literals once a
    deployment's lanes are known. The render below is the single-lane shape —
    every project that has not opted into a second lane — which is the shape
    these names have always had.
    """

    #: What ``resolve_limits_mount`` hands the template: the two halves of the
    #: limits bind, already anchored host-side. Spelled here only so the test
    #: can assert the template pastes them through verbatim.
    LIMITS_MOUNT = {
        "source": "./data/channel_limits.json",
        "target": "/app/project/data/channel_limits.json",
    }

    @classmethod
    def _template_text(cls, **extra):
        """The single-lane ``full`` render, optionally with staging context.

        The pinned lane contexts carry the shape the injector produces and
        stop there; the two keys the deploy adds at render time are supplied
        here. ``limits_mount`` unconditionally, because the ``full`` shape
        enables writes and the template hard-fails without it (its halves are
        host-computed, so there is no template-side default to fall back on).
        ``bluesky_devices`` only when a caller asks: its absence is the
        browse-only shape, which the gate test below renders on purpose.
        """
        from tests.deployment.test_lane_compose import _render_text, _single_lane_contexts

        context = dict(_single_lane_contexts()["full"])
        context["limits_mount"] = cls.LIMITS_MOUNT
        context.update(extra)
        return _render_text(context)

    def test_every_expansion_of_both_halves_is_guarded_fail_closed(self):
        """`:?` on BOTH names is why an empty value of either is refused upstream.

        Asserted as "no UNGUARDED expansion exists" rather than as a count: the
        public key is consumed by more than one service (the queueserver needs
        it for its own `qserver ping` healthcheck handshake), and a future
        consumer added without the guard is exactly the regression worth
        catching. A count would also break on any benign re-plumbing.
        """
        import re

        text = self._template_text()

        for var in ("BLUESKY_QSERVER_ZMQ_PRIVATE_KEY", "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY"):
            assert f"${{{var}:?" in text, f"{var} must carry a fail-closed `:?` guard"
            unguarded = re.findall(rf"\$\{{{var}(?![:?])", text)
            assert not unguarded, f"{var} has {len(unguarded)} expansion(s) with no `:?` guard"

    def test_document_plane_paths_are_not_env_passthroughs(self):
        """The certificate paths are hardcoded, so nothing derives them into .env.

        This is the contract that makes ``_ensure_bluesky_document_plane_certs``
        generate *files* rather than environment variables. If the template ever
        switches these to ``${...}`` passthroughs, this module has to change too.
        """
        text = self._template_text()

        assert "BLUESKY_ZMQ_CURVE_SECRET_KEY: /app/curve/proxy.key_secret" in text
        assert "BLUESKY_ZMQ_CURVE_CLIENT_PUBLIC_KEYS: /app/curve/clients" in text
        assert "${BLUESKY_ZMQ_CURVE_SECRET_KEY" not in text

    def test_certificate_mount_sources_match_the_generated_layout(self):
        """The bind sources are the directories this module writes."""
        text = self._template_text()

        assert "./data/.runtime/bluesky_curve/bridge:/app/curve:ro" in text
        assert "./data/.runtime/bluesky_curve/queueserver:/app/curve:ro" in text

    def test_device_file_env_and_mount_are_the_staged_pair(self):
        """The worker's device variable names exactly what the bind carries.

        The two halves are the whole contract between the render-time staging
        step and the container: ``_stage_bluesky_devices`` copies the authored
        document into the bluesky service's build context under a fixed
        basename, and the worker switches on ``BLUESKY_DEVICES_FILE`` alone. A
        change to either the staged path or the container path that is not
        matched on the other side leaves the worker naming a file the mount
        does not carry, so the two are pinned together rather than separately.
        """
        text = self._template_text(bluesky_devices=True)

        assert "BLUESKY_DEVICES_FILE: /app/project/data/bluesky_devices.yml" in text
        assert (
            "./build/services/bluesky/bluesky_devices.yml:/app/project/data/bluesky_devices.yml:ro"
        ) in text

    def test_device_wiring_is_absent_when_nothing_was_staged(self):
        """No staged file means no variable and no mount -- fail closed.

        The browse-only shape. Naming a file the bind does not carry would
        fail the worker's own load, which is why the gate defaults false
        rather than pointing at a path and hoping.
        """
        text = self._template_text()

        assert "BLUESKY_DEVICES_FILE" not in text
        assert "bluesky_devices.yml" not in text

    def test_limits_mount_halves_are_pasted_through_verbatim(self):
        """Both halves of the limits bind are computed host-side, not spelled here.

        ``resolve_limits_mount`` anchors the source against the compose project
        directory and the target against the container's project root; a
        template that derived either would be a second spelling of a rule the
        connector already follows. Rendered on both the bridge and the
        queueserver, because the reference monitor moved into the worker and
        the limits DB had to follow it there.
        """
        source = self.LIMITS_MOUNT["source"]
        target = self.LIMITS_MOUNT["target"]

        text = self._template_text()

        assert text.count(f"- {source}:{target}:ro") == 2

    def test_limits_mount_is_gated_on_writes(self):
        """A read-only deploy never opens the limits DB, so it never mounts it."""
        from tests.deployment.test_lane_compose import _render_text, _single_lane_contexts

        context = dict(_single_lane_contexts()["minimal"])
        context["limits_mount"] = self.LIMITS_MOUNT

        assert self.LIMITS_MOUNT["source"] not in _render_text(context)


class TestGeneratedProjectGitignore:
    """The scaffolded project must actually ignore the certificate directory.

    Asserted through ``git check-ignore`` rather than by reading the file,
    because the thing that matters is git's behavior, not the text.
    """

    @staticmethod
    def _repo_with_template_gitignore(tmp_path):
        """A git repo whose .gitignore is the project scaffolding template."""
        import subprocess

        import osprey.templates

        # Resolved off the templates package root, the same way scaffolding.py
        # reaches it — `osprey.templates.project` is a plain data directory,
        # not an importable module with a __file__.
        template = Path(osprey.templates.__file__).parent / "project" / "gitignore"
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        (tmp_path / ".gitignore").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
        return tmp_path

    @staticmethod
    def _is_ignored(repo, relative_path):
        import subprocess

        return (
            subprocess.run(
                ["git", "check-ignore", "-q", "--no-index", str(relative_path)],
                cwd=repo,
                capture_output=True,
            ).returncode
            == 0
        )

    def test_certificate_directory_is_ignored(self, tmp_path):
        repo = self._repo_with_template_gitignore(tmp_path)

        for role in ("bridge/proxy.key_secret", "queueserver/publisher.key_secret"):
            assert self._is_ignored(repo, Path("data/.runtime/bluesky_curve") / role), role

    def test_channel_limits_stays_tracked(self, tmp_path):
        """The entry must not swallow the rest of data/ — channel_limits.json is committed."""
        repo = self._repo_with_template_gitignore(tmp_path)

        assert not self._is_ignored(repo, Path("data/channel_limits.json"))

    def test_pattern_is_anchored_to_the_project_root(self, tmp_path):
        """An unanchored pattern would silently swallow like-named paths elsewhere.

        This repo has been bitten before by unanchored ignore rules making
        ``git add`` skip a moved file with no error, so the anchoring is the
        point of the test, not an incidental detail.
        """
        repo = self._repo_with_template_gitignore(tmp_path)

        assert not self._is_ignored(repo, Path("src/data/.runtime/bluesky_curve/proxy.key_secret"))
        assert not self._is_ignored(repo, Path("tests/fixtures/.runtime/proxy.key_secret"))


class TestDeployWiring:
    """Provisioning runs on the deploy paths that need it, before compose."""

    def test_deploy_up_calls_both_provisioners(self):
        """Both provisioners run before the image build, on every start path.

        Read off ``_start_stack`` rather than ``deploy_up``: the deploy sequence
        moved there when ``deploy_up`` (render, then start) and ``up_as_built``
        (start what a build already rendered) were made to share one. The
        property being pinned is the ORDER of that sequence, so it belongs to
        the function that has one — and the last assertion keeps every start
        verb answerable to it.

        The repo verbs reach it one hop away, through ``_start_as_built``: the
        start half was split out of ``up_as_built`` so ``restart_deployment``
        could run a ``down`` between the read and the start. A verb that grew a
        third way to start would fail here, which is the point.
        """
        import inspect

        from osprey.deployment import container_lifecycle

        source = inspect.getsource(container_lifecycle._start_stack)
        assert "_ensure_bluesky_control_plane_keys(config" in source
        assert "_ensure_bluesky_document_plane_certs(config" in source
        # Key material has to exist before compose mounts it: a bind source
        # that does not exist is created root-owned by the runtime.
        assert source.index("_ensure_bluesky_document_plane_certs(config") < source.index(
            "_build_project_image(config, dev_mode, env"
        )
        assert "_start_stack(" in inspect.getsource(container_lifecycle._start_as_built)
        for verb in (
            container_lifecycle.deploy_up,
            container_lifecycle.up_as_built,
            container_lifecycle.restart_deployment,
        ):
            body = inspect.getsource(verb)
            assert "_start_stack(" in body or "_start_as_built(" in body

    def test_deploy_restart_mints_the_control_plane_pair(self):
        """The template's `:?` guard aborts a restart on an unset private key.

        A project whose ``.env`` predates this feature would otherwise be
        broken by the very act of restarting it — the same reason
        ``_ensure_service_tokens`` already runs on this path.
        """
        import inspect

        from osprey.deployment import container_lifecycle

        source = inspect.getsource(container_lifecycle.deploy_restart)
        # Matched on the call NAME, not on a full argument list: the call is
        # anchored on the repo root's `.env` (it mints a keypair, and a
        # cwd-relative default would strand it), and pinning the exact argument
        # text would make this test fail on the anchoring rather than on the
        # property it is here to protect — that the mint happens before restart.
        assert "_ensure_bluesky_control_plane_keys(" in source
        assert source.index("_ensure_bluesky_control_plane_keys(") < source.index('"restart"')

    def test_deploy_restart_does_not_generate_certificates(self):
        """`compose restart` reuses existing container config; mounts are not re-resolved."""
        import inspect

        from osprey.deployment import container_lifecycle

        source = inspect.getsource(container_lifecycle.deploy_restart)
        assert "_ensure_bluesky_document_plane_certs(" not in source


def test_default_env_path_is_cwd_relative(tmp_path, monkeypatch):
    """Both new functions follow ``_ensure_service_tokens``' CWD convention.

    A caller that passes no ``env_path`` gets a default ``Path(".env")``, which
    resolves against whatever directory the process is in — which is why every
    live path passes the repo's own ``.env`` explicitly instead.
    """
    monkeypatch.delenv("BLUESKY_QSERVER_ZMQ_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("BLUESKY_QSERVER_ZMQ_PUBLIC_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    config = {"deployed_services": ["bluesky"]}

    _ensure_bluesky_control_plane_keys(config)
    _ensure_bluesky_document_plane_certs(config)

    assert (tmp_path / ".env").is_file()
    assert (
        tmp_path / "data" / ".runtime" / "bluesky_curve" / "bridge" / "proxy.key_secret"
    ).is_file()
    assert os.environ.get("BLUESKY_QSERVER_ZMQ_PRIVATE_KEY") is None  # never exported


# ---------------------------------------------------------------------------
# Docker-compose interpolation safety.
#
# `docker compose --env-file` INTERPOLATES the file it reads: `$D` in a value is
# a variable reference, and an unset variable expands to the empty string. The
# Z85 alphabet the RE manager's CURVE keys use includes `$`, so a valid minted
# key can reach the container two characters shorter and be rejected as "not a
# 40 byte z85 encoded string" -- intermittently, since ~62% of keys contain a
# `$` and it only corrupts when the next character starts an identifier.
#
# Not hypothetical: a real `osprey up` wrote `...CRx[$D(IbLK...` and the
# bridge received `...CRx[(IbLK...`, failing every route that touches the
# manager. Found by tests/e2e/test_bluesky_queue_e2e.py. NOTHING at this layer
# could have seen it before, because every test here reads the file back with
# osprey's own parser, which does not interpolate -- so the value on disk always
# looked correct.
# ---------------------------------------------------------------------------


def _interpolate_like_compose(value: str, environment: dict[str, str] | None = None) -> str:
    """Apply docker compose's env-file interpolation to one value.

    Models the two forms that matter -- ``$NAME`` and ``${NAME}`` -- with unset
    names expanding to the empty string, which is compose's documented
    behaviour and exactly the mechanism that corrupted a real key. Modelled
    here rather than shelled out to ``docker compose config`` so this stays a
    unit test with no daemon; the calibration test below keeps the model
    honest against the real observed failure.
    """
    environment = environment or {}

    def _expand(match: re.Match) -> str:
        return environment.get(match.group("braced") or match.group("bare"), "")

    return re.sub(r"\$(?:\{(?P<braced>\w+)\}|(?P<bare>\w+))", _expand, value)


def test_the_interpolation_model_corrupts_the_real_observed_key() -> None:
    """Calibration for the model above, against the key from the real failure.

    Without it, `_interpolate_like_compose` could quietly stop expanding
    anything and every assertion below would pass vacuously.
    """
    observed = "Go0c&>CUqjKpYCRx[$D(IbLKO>DW<o>&gM>JyFzm"
    assert len(observed) == 40
    corrupted = _interpolate_like_compose(observed)
    assert corrupted == "Go0c&>CUqjKpYCRx[(IbLKO>DW<o>&gM>JyFzm"
    assert len(corrupted) == 38


class TestControlPlaneKeysSurviveComposeInterpolation:
    def test_minted_keys_round_trip_through_interpolation_unchanged(self, env_path) -> None:
        """The pin that would have caught this: every minted value must survive
        interpolation byte-for-byte.

        Repeated, because the defect is probabilistic -- a single mint passes
        by luck ~38% of the time, which is worse than no test at all.
        """
        for _ in range(40):
            env_path.unlink(missing_ok=True)
            _ensure_bluesky_control_plane_keys(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )
            written = _dotenv(env_path)
            for var in ("BLUESKY_QSERVER_ZMQ_PRIVATE_KEY", "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY"):
                value = written[var]
                assert len(value) == 40, f"{var} is not a 40-character z85 key: {value!r}"
                assert _interpolate_like_compose(value) == value, (
                    f"{var} would be rewritten by docker compose on its way to the "
                    f"container: {value!r}"
                )

    def test_an_operator_supplied_key_containing_a_dollar_is_refused_loudly(self, env_path) -> None:
        """Existing values are never overwritten, so refusing is the only honest
        option: minting over a key the operator deliberately set would be worse
        than failing, and the compose ``:?`` guard cannot catch a value that IS
        set -- it would surface as an opaque control-socket timeout instead.
        """
        env_path.write_text(
            "BLUESKY_QSERVER_ZMQ_PRIVATE_KEY=R7uS%hYUbbCxSjwn::04BS$Dcim0^D{41L/Y)brp(\n",
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError) as excinfo:
            _ensure_bluesky_control_plane_keys(
                {"deployed_services": ["bluesky"]}, env_path=env_path
            )

        message = str(excinfo.value)
        # Names the variable AND the mechanism -- an operator cannot act on
        # "invalid key" alone.
        assert "BLUESKY_QSERVER_ZMQ_PRIVATE_KEY" in message
        assert "docker compose" in message
        assert "$" in message

    def test_every_other_minted_secret_uses_a_dollar_free_alphabet(self) -> None:
        """The audit, as a test rather than a note.

        Every other secret this deploy path mints draws from an alphabet that
        cannot emit ``$``: ``token_urlsafe`` (``[A-Za-z0-9_-]``), ``token_hex``
        (``[0-9a-f]``), and OpenObserve's recipe, whose own docstring already
        excludes ``$`` deliberately. The manager's Z85 keys were the single
        exception. This keeps that negative result from going stale the next
        time a secret is added with a hand-rolled alphabet.
        """
        from osprey.deployment.service_tokens import _VAR_GENERATORS, _default_token

        recipes = {"<default>": _default_token, **_VAR_GENERATORS}
        for name, recipe in recipes.items():
            for _ in range(20):
                assert "$" not in recipe(), (
                    f"{name} can mint a value containing '$', which docker compose would "
                    "rewrite out of the project .env"
                )
