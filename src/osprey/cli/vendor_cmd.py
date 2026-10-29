"""CLI commands for managing vendor assets (JS/CSS/fonts)."""

import click

from .output import fail, report


@click.group("vendor")
def vendor():
    """Manage locally bundled vendor assets (JS/CSS/fonts).

    By default OSPREY interfaces load third-party libraries directly from CDN,
    so no setup is required. These commands populate static/vendor/ dirs
    for firewalled deployments — set OSPREY_OFFLINE=1 (or 'offline: true' in
    config.yml) to switch the interfaces over to the local bundles.
    """


@vendor.command()
@click.option("--quiet", "-q", is_flag=True, help="Suppress per-file output.")
@click.option(
    "--insecure",
    "-k",
    is_flag=True,
    help=(
        "Skip TLS cert verification. Safe — every asset is checked against "
        "its manifest SHA256. Use behind corporate proxies (e.g. Squid) that "
        "intercept TLS. Also enabled by OSPREY_VENDOR_INSECURE=1."
    ),
)
@click.option(
    "--target",
    "-t",
    "targets",
    multiple=True,
    metavar="DIR",
    help=(
        "Fetch only the assets for this manifest target directory (repeatable), "
        "e.g. tuning/static/vendor/. Default: every target."
    ),
)
def fetch(quiet: bool, insecure: bool, targets: tuple[str, ...]) -> None:
    """Download the vendor assets declared in the manifest.

    Run this once on firewalled deployments before starting 'osprey web'
    with OSPREY_OFFLINE=1. In default CDN mode this command is optional,
    except for package-based targets a panel always loads locally, which an
    image fetches with --target.
    """
    from osprey.interfaces.vendor import fetch_all

    if not quiet:
        report("Fetching vendor assets...")
    try:
        downloaded = fetch_all(quiet=quiet, insecure=insecure, targets=targets or None)
    except (RuntimeError, ValueError) as exc:
        fail("Could not fetch the vendor assets", str(exc))
        raise SystemExit(1) from exc
    if downloaded:
        report(f"Downloaded {len(downloaded)} file(s).")
    else:
        report("All vendor assets already up to date.")


@vendor.command()
def verify() -> None:
    """Verify all vendor assets exist with correct checksums."""
    from osprey.interfaces.vendor import verify_all

    ok, problems = verify_all()
    if problems:
        fail(
            f"{len(problems)} vendor file(s) did not verify",
            "\n".join(str(p) for p in problems),
            "Download them again with `osprey vendor fetch`.",
        )
        raise SystemExit(1)
    report(f"All {len(ok)} vendor files verified OK.")
