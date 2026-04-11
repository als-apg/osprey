"""CLI commands for managing vendor assets (JS/CSS/fonts)."""

import click


@click.group("vendor")
def vendor():
    """Manage vendor assets (JS/CSS/fonts).

    Vendor assets are third-party libraries declared in vendor_manifest.json.
    They are fetched at build time rather than tracked in git or loaded from CDN.
    """


@vendor.command()
@click.option("--quiet", "-q", is_flag=True, help="Suppress per-file output.")
def fetch(quiet: bool) -> None:
    """Download all vendor assets declared in the manifest."""
    from osprey.interfaces.vendor import fetch_all

    if not quiet:
        click.echo("Fetching vendor assets...")
    downloaded = fetch_all(quiet=quiet)
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
