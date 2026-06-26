"""CLI commands for managing vendor assets (JS/CSS/fonts)."""

import click


@click.group("vendor")
def vendor():
    """Manage locally bundled vendor assets (JS/CSS/fonts).

    By default OSPREY interfaces load third-party libraries directly from CDN,
    so no setup is required. These commands populate ``static/vendor/`` dirs
    for firewalled deployments — set ``OSPREY_OFFLINE=1`` (or
    ``offline: true`` in ``config.yml``) to switch the interfaces over to the
    local bundles.
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
def fetch(quiet: bool, insecure: bool) -> None:
    """Download all vendor assets declared in the manifest.

    Run this once on firewalled deployments before starting ``osprey web``
    with ``OSPREY_OFFLINE=1``. In default CDN mode this command is optional.
    """
    from osprey.interfaces.vendor import fetch_all

    if not quiet:
        click.echo("Fetching vendor assets...")
    try:
        downloaded = fetch_all(quiet=quiet, insecure=insecure)
    except RuntimeError as exc:
        click.echo(f"\n{exc}", err=True)
        raise SystemExit(1) from exc
    if downloaded:
        click.echo(f"Downloaded {len(downloaded)} file(s).")
    else:
        click.echo("All vendor assets already up to date.")


@vendor.command()
def verify() -> None:
    """Verify all vendor assets exist with correct checksums."""
    from osprey.interfaces.vendor import verify_all

    ok, problems = verify_all()
    if problems:
        click.echo(f"{len(problems)} problem(s) found:", err=True)
        for p in problems:
            click.echo(f"  {p}", err=True)
        raise SystemExit(1)
    click.echo(f"All {len(ok)} vendor files verified OK.")
