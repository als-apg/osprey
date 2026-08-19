"""The sidecar's other model-delivery path, against a real container.

By default the qmd image bakes its three GGUF models in at build time, which
needs a build host that can reach huggingface.co. A host that cannot gets the
other path: ``services.qmd.models_dir`` names a host directory holding the same
three files, the build is told to skip the downloads
(``--build-arg OSPREY_QMD_MODELS_MOUNTED=1``), and the files arrive at runtime
over a read-only bind mount instead.

That path has one failure mode worth this whole module: **the models are not
there**. Nothing about it is loud on its own. qmd's reaction to a model file it
cannot make sense of is to download a replacement, on a host chosen for this
path precisely because it has no route out, under a compose health check that
allows an hour of start-up. So the entrypoint checks the staged files against
the digests the image was built with, before it starts anything, and refuses
with staging instructions when they do not hold up. These cases are that
refusal and its opposite, run for real.

The image under test is the skip-ARG build, named by ``OSPREY_QMD_MOUNTED_IMAGE``
— the *only* image the mounted path can be tested against, since a baked one
already carries the models. It is cheap to build for exactly the reason this
feature exists: it downloads none of the 2.1 GB.

The staged files are kilobyte stand-ins. The check under test is a name, a
non-zero size and a SHA256 comparison; the expected digests are supplied to the
container as ``OSPREY_QMD_*_SHA256``, which is what the image's own build would
have baked in. Staging 2.1 GB of real models in CI to assert a digest
comparison would prove nothing extra and cost the download this path exists to
avoid.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from tests._container_support import is_docker_available
from tests.integration._qmd_ariel_support import (
    MODEL_DIGEST_ENV,
    SIDECAR_MODEL_DIR,
    require_mounted_models_image,
    sidecar_logs,
    sidecar_over_models_dir,
    stage_model_fixtures,
    wait_for_sidecar_exit,
    wait_for_sidecar_log,
)

# xdist_group("docker") pins every container-starting module onto one worker, so
# a run has a single testcontainers session and a single ryuk reaper.
pytestmark = [
    pytest.mark.dockerbuild,
    pytest.mark.integration,
    pytest.mark.xdist_group("docker"),
]

#: The model the entrypoint reaches first, and so the one an empty directory is
#: reported against. Its digest variable leads ``MODEL_DIGEST_ENV`` for the same
#: reason: the order is the load order.
EMBED_MODEL = next(iter(MODEL_DIGEST_ENV))

#: The verdict line the model check ends on when every staged file holds up.
#: The delivery word in it is the image's own ``OSPREY_QMD_MODEL_DELIVERY``, so
#: a baked image logging this would be reporting the wrong path.
VERIFIED = "models verified against the digests this image was built with (mounted delivery)"

#: Every line the entrypoint writes carries this tag after its timestamp.
#: Asserting through it is what makes an indent assertion mean the indent of a
#: message rather than of some substring inside one.
TAG = "[qmd-sidecar] "


def refusal_logs(container) -> str:
    """The complete output of a container that refused to start.

    Waits for the FATAL line first, so a container that hangs instead of
    refusing fails with its own logs rather than with a bare timeout — then
    waits for the exit before reading, because the refusal is many lines long
    and a read taken while it is still being written returns a message that
    looks truncated when it is only unfinished.
    """
    wait_for_sidecar_log(container, "FATAL: model file")
    assert wait_for_sidecar_exit(container) != 0, "the sidecar logged a refusal and then exited 0"
    return sidecar_logs(container)


@pytest.fixture
def mounted_image() -> str:
    """The skip-ARG image, or a skip/failure per its build state."""
    if not is_docker_available():
        pytest.skip("docker daemon is not reachable")
    return require_mounted_models_image()


@pytest.fixture
def state_volume():
    """A named volume standing in for the deployment's ``qmd_index``.

    Named uniquely per test and removed afterwards — never a prune, matching
    the container-ops guardrail the rest of the suite observes.
    """
    import docker

    client = docker.from_env()
    volume = client.volumes.create(name=f"osprey-qmd-models-test-{uuid.uuid4().hex[:8]}")
    try:
        yield volume.name
    finally:
        try:
            volume.remove(force=True)
        except Exception:  # pragma: no cover - teardown best effort
            pass


def test_an_empty_mount_refuses_and_says_what_to_stage(mounted_image: str, tmp_path: Path):
    """The whole point of the pre-start check, in the state that motivated it.

    An operator who configured ``models_dir`` and never staged the files gets a
    named file, a named cause, the directory, and the three exact names — not an
    hour of a container that looks like it is starting.
    """
    with sidecar_over_models_dir(mounted_image, tmp_path) as container:
        logs = refusal_logs(container)

        assert (
            f"FATAL: model file '{EMBED_MODEL}' cannot be used: "
            f"no such file in {SIDECAR_MODEL_DIR}" in logs
        )
        assert f"Stage all three model files in {SIDECAR_MODEL_DIR}, named exactly:" in logs
        for name in MODEL_DIGEST_ENV:
            assert f"{TAG}    {name}" in logs
        # The image knows its models come from the mount, so the remedy is about
        # the host directory rather than about rebuilding the image.
        assert "This image expects the models at runtime, so stage them under those" in logs
        assert f"names in the host directory bind-mounted at {SIDECAR_MODEL_DIR}." in logs


def test_staged_models_verify_against_the_digests_the_image_records(
    mounted_image: str, tmp_path: Path
):
    """The path a correctly-staged host takes: three files in, verification out.

    Asserted at the verification line rather than at ``/health``, because
    everything after it — the index pass, the daemon, the forwarder — belongs to
    the sidecar suites that already cover it over a real corpus. What is new
    here is that a *mounted* image accepts files it did not bake.
    """
    digests = stage_model_fixtures(tmp_path)

    with sidecar_over_models_dir(mounted_image, tmp_path, env=digests) as container:
        before = wait_for_sidecar_log(container, VERIFIED).split(VERIFIED)[0]

        # Nothing was refused on the way there. Read up to the verification line
        # rather than over the whole log: the container carries on into the
        # startup pass, which fails closed on an empty index because this case
        # stages models and no corpus — a real refusal, about something else.
        assert "FATAL" not in before
        # Cold start: no stamp yet, so every file is actually hashed. 1024 is
        # the fixture size, so this also proves the check read the staged file
        # rather than something already in the image.
        assert f"hashing {EMBED_MODEL} (1024 bytes)" in before


def test_a_staged_file_that_is_not_the_expected_one_refuses(mounted_image: str, tmp_path: Path):
    """Right name, wrong bytes — the case a name-only check would wave through.

    A truncated copy or a file staged from a different model release reads as
    present to everything except the digest, and qmd's own reaction to loading
    it is a crash or a silent re-download rather than a message.
    """
    digests = stage_model_fixtures(tmp_path)
    on_disk = digests[MODEL_DIGEST_ENV[EMBED_MODEL]]
    # Point the embed model's expected digest at a different staged file, which
    # is the shape of a mis-staged directory without needing a second fixture.
    expected = digests["OSPREY_QMD_RERANK_SHA256"]
    env = {**digests, MODEL_DIGEST_ENV[EMBED_MODEL]: expected}

    with sidecar_over_models_dir(mounted_image, tmp_path, env=env) as container:
        logs = refusal_logs(container)

        assert (
            f"FATAL: model file '{EMBED_MODEL}' cannot be used: SHA256 mismatch: "
            f"this image expects {expected}, the file on disk is {on_disk}" in logs
        )


def test_a_restart_trusts_the_stamp_instead_of_rehashing(
    mounted_image: str, tmp_path: Path, state_volume: str
):
    """Verification is once per staged file, not once per start.

    Hashing 2.1 GB of real models costs about half a minute, which is fine on a
    first start and pure delay on every restart after it. The stamp in the state
    volume is what turns it into the former; its absence would be invisible
    except as a container that takes longer than it should to come back.
    """
    digests = stage_model_fixtures(tmp_path)

    with sidecar_over_models_dir(
        mounted_image, tmp_path, env=digests, state_volume=state_volume
    ) as first:
        assert f"hashing {EMBED_MODEL}" in wait_for_sidecar_log(first, VERIFIED)

    with sidecar_over_models_dir(
        mounted_image, tmp_path, env=digests, state_volume=state_volume
    ) as second:
        logs = wait_for_sidecar_log(second, VERIFIED).split(VERIFIED)[0]

    assert "hashing" not in logs, (
        "the second start re-hashed the staged models, so the stamp written by the "
        f"first start was not read back:\n{logs}"
    )


def test_the_staged_names_are_the_names_the_deploy_preflight_checks():
    """The host-side refusal and the container-side one have to name the same
    three files, or a deploy passes its preflight and the container refuses
    anyway (or worse, the other way round). No container needed."""
    from osprey.deployment.qmd_service import MODEL_FILENAMES

    assert tuple(MODEL_DIGEST_ENV) == MODEL_FILENAMES


def test_the_mount_target_is_the_directory_the_image_reads(mounted_image: str):
    """The bind target is a literal on both sides — a rename of the image's
    ``OSPREY_QMD_MODEL_DIR`` that missed this constant would mount the models
    somewhere the container never looks, and the deployment would report the
    staged files as missing."""
    import docker

    image = docker.from_env().images.get(mounted_image)
    env = dict(item.split("=", 1) for item in image.attrs["Config"]["Env"])

    assert env["OSPREY_QMD_MODEL_DIR"] == SIDECAR_MODEL_DIR
    # And the image knows it is the mounted build, which is what selects the
    # staging half of the refusal text asserted above.
    assert env["OSPREY_QMD_MODEL_DELIVERY"] == "mounted"
