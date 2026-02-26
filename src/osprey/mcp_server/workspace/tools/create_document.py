"""MCP tool: create_document — compile LaTeX source to PDF.

Handles both Beamer slides and article reports. Resolves artifact IDs
to include figures in the build directory. Produces .tex source and
.pdf output as separate artifacts.
"""

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from osprey.mcp_server.common import make_error
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.create_document")


def _resolve_artifacts_to_build_dir(artifact_ids: list[str], build_dir: Path) -> list[str]:
    """Copy artifact files into the build directory for LaTeX \\includegraphics.

    Returns list of filenames available in build_dir (sans ID prefix).
    """
    from osprey.mcp_server.artifact_store import get_artifact_store

    store = get_artifact_store()
    available: list[str] = []

    for art_id in artifact_ids:
        art_path = store.get_file_path(art_id)
        if art_path is None or not art_path.exists():
            logger.warning("Artifact %s not found, skipping", art_id)
            continue

        # Strip the artifact ID prefix (e.g., "a1b2c3d4e5f6_figure.png" -> "figure.png")
        original_name = art_path.name
        parts = original_name.split("_", 1)
        clean_name = parts[1] if len(parts) > 1 else original_name

        dest = build_dir / clean_name
        shutil.copy2(art_path, dest)
        available.append(clean_name)

    return available


@mcp.tool()
async def create_document(
    latex_source: str,
    title: str,
    description: str = "",
    artifact_ids: list[str] | None = None,
) -> str:
    """Compile LaTeX source to PDF. Supports Beamer slides and article reports.

    Runs pdflatex (2 passes for cross-references) on the provided LaTeX source.
    Both the .tex source and compiled .pdf are saved as artifacts.

    To include figures, pass their artifact IDs in the artifact_ids parameter.
    The files are copied to the build directory with their original filenames
    (sans ID prefix), so reference them with \\includegraphics{filename.png}.

    Args:
        latex_source: Complete LaTeX source code.
        title: Document title for artifact metadata.
        description: Description of the document.
        artifact_ids: Optional list of artifact IDs of figures to include
            in the build directory.

    Returns:
        JSON with artifact_ids (PDF + source) and context_entry_id.
    """
    if not latex_source or not latex_source.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "No LaTeX source provided.",
                ["Provide complete LaTeX source code."],
            )
        )

    # Check pdflatex availability
    if shutil.which("pdflatex") is None:
        return json.dumps(
            make_error(
                "dependency_missing",
                "pdflatex is not installed.",
                [
                    "macOS: brew install --cask mactex  (or: brew install basictex)",
                    "Ubuntu: apt install texlive-latex-base texlive-latex-extra",
                    "pdflatex must be installed as a system package, not via pip.",
                ],
            )
        )

    with tempfile.TemporaryDirectory(prefix="osprey_latex_") as tmp:
        build_dir = Path(tmp)

        # Copy referenced artifacts into build dir
        resolved_files: list[str] = []
        if artifact_ids:
            resolved_files = _resolve_artifacts_to_build_dir(artifact_ids, build_dir)

        # Write LaTeX source
        tex_path = build_dir / "document.tex"
        tex_path.write_text(latex_source, encoding="utf-8")

        # Run pdflatex (2 passes for cross-references, TOC, etc.)
        compile_errors: list[str] = []
        for pass_num in range(1, 3):
            try:
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "document.tex"],
                    cwd=str(build_dir),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    # Extract meaningful error lines from pdflatex output
                    error_lines = [
                        line
                        for line in result.stdout.splitlines()
                        if line.startswith("!") or "Error" in line
                    ]
                    compile_errors = error_lines or [result.stdout[-2000:]]
            except subprocess.TimeoutExpired:
                return json.dumps(
                    make_error(
                        "execution_error",
                        f"pdflatex timed out on pass {pass_num}.",
                        ["Simplify the LaTeX source or check for infinite loops."],
                    )
                )

        # Check if PDF was produced
        pdf_path = build_dir / "document.pdf"
        if not pdf_path.exists():
            return json.dumps(
                make_error(
                    "compilation_error",
                    "LaTeX compilation failed — no PDF produced.",
                    compile_errors[:5] or ["Check the LaTeX source for syntax errors."],
                )
            )

        # Save artifacts
        from osprey.mcp_server.artifact_store import get_artifact_store

        store = get_artifact_store()
        output_artifact_ids: list[str] = []

        # Save PDF
        pdf_entry = store.save_file(
            file_content=pdf_path.read_bytes(),
            filename=f"{title[:40].replace(' ', '_')}.pdf",
            artifact_type="file",
            title=title,
            description=description or f"PDF document: {title}",
            mime_type="application/pdf",
            tool_source="create_document",
            metadata={"figures_included": len(resolved_files)},
        )
        pdf_entry.category = "document"
        pdf_entry.source_agent = "data-visualizer"
        output_artifact_ids.append(pdf_entry.id)

        # Save .tex source
        tex_entry = store.save_file(
            file_content=latex_source.encode("utf-8"),
            filename=f"{title[:40].replace(' ', '_')}.tex",
            artifact_type="text",
            title=f"Source: {title}",
            description=f"LaTeX source for: {title}",
            mime_type="application/x-tex",
            tool_source="create_document",
        )
        tex_entry.category = "document"
        tex_entry.source_agent = "data-visualizer"
        output_artifact_ids.append(tex_entry.id)

        store._save_index()

    response: dict = {
        "status": "success",
        "title": title,
        "artifact_ids": output_artifact_ids,
        "pdf_artifact_id": output_artifact_ids[0],
        "source_artifact_id": output_artifact_ids[1],
        "figures_included": len(resolved_files),
    }
    try:
        from osprey.mcp_server.common import gallery_url

        response["gallery_url"] = gallery_url()
    except Exception:
        pass
    return json.dumps(response, default=str)
