#!/usr/bin/env python3
"""Render the versioned execution plan and record its exact build inputs.

This is a CPU documentation build. It reuses build_paper.py without changing
that renderer or rebuilding the proposal. Installed dependencies are Pandoc,
ReportLab, matplotlib, Pillow, pypdf and Liberation fonts. It installs nothing.

The source supplies Document ID, Version, Date and Status metadata. Every build
resets visual_review to pending: successful rendering is not visual inspection
or approval by the project owner.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import platform
import re
import subprocess


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = ROOT / "execution_plan.md"
OUTPUT = ROOT / "execution_plan.pdf"
MANIFEST = ROOT / "execution_plan.build.json"
RENDERER = ROOT / "build_paper.py"
CACHE = ROOT / ".build" / "execution_plan"

spec = util.spec_from_file_location("adaptive_plan_renderer", RENDERER)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load the shared document renderer")
base = util.module_from_spec(spec)
spec.loader.exec_module(base)
base.CACHE = CACHE
CACHE.mkdir(parents=True, exist_ok=True)
base.styles = {name: copy.deepcopy(style) for name, style in base.styles.items()}
base.styles["title"].fontSize = 22
base.styles["title"].leading = 25.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def source_metadata(text: str) -> dict[str, str]:
    """Require explicit metadata; do not infer a release or approval state."""
    plain = text.replace("**", "")
    fields = {}
    for label, key in (("Document ID", "document_id"), ("Version", "version"),
                       ("Date", "date"), ("Status", "status")):
        match = re.search(rf"^{re.escape(label)}:\s*(.+?)\s*$", plain, re.M)
        if not match:
            raise ValueError(f"execution_plan.md needs a '{label}:' metadata line")
        fields[key] = match.group(1).strip()
    if fields["document_id"] != "OAA-PLAN":
        raise ValueError("Expected document ID OAA-PLAN")
    if not re.fullmatch(r"\d+\.\d+\.\d+", fields["version"]):
        raise ValueError("Version must contain three numeric components")
    title = re.search(r"^#\s+(.+?)\s*$", text, re.M)
    if not title:
        raise ValueError("execution_plan.md needs a top-level title")
    fields["title"] = title.group(1)
    return fields


class ExecutionPlanDoc(base.PaperDoc):
    def __init__(self, path: str, info: dict[str, str], source_hash: str):
        self.info = info
        base.BaseDocTemplate.__init__(
            self, path, pagesize=base.letter,
            leftMargin=42, rightMargin=42, topMargin=43, bottomMargin=42,
            title=info["title"], author="Prasuna Chatla",
            subject=(f'{info["document_id"]} {info["version"]}; '
                     f'{info["status"]}; source_sha256={source_hash}'),
        )
        frame = base.Frame(42, 42, 528, 707, leftPadding=0, rightPadding=0,
                           topPadding=0, bottomPadding=0)
        self.addPageTemplates(base.PageTemplate(
            id="body", frames=[frame], onPage=self.chrome))

    def chrome(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(base.MUTED)
        canvas.setFont("Body", 7.6)
        canvas.drawString(42, 768, "ONLINE ACTUATOR ADAPTATION  |  Execution plan")
        canvas.drawRightString(570, 768,
                              f'{self.info["document_id"]}  {self.info["version"]}')
        canvas.setStrokeColor(base.LINE)
        canvas.setLineWidth(.5)
        canvas.line(42, 759, 570, 759)
        canvas.line(42, 32, 570, 32)
        canvas.setFont("Body", 7.8)
        canvas.drawString(42, 20, f'Prasuna Chatla  |  {self.info["date"]}')
        canvas.drawRightString(570, 20, str(doc.page))
        canvas.restoreState()


def image_inputs(node) -> set[Path]:
    """Include repository-local figure bytes if a later plan adds a figure."""
    found = set()
    if isinstance(node, dict):
        if node.get("t") == "Image":
            path = ROOT / node["c"][2][0]
            if path.exists():
                found.add(path)
            if path.suffix == ".svg" and path.with_suffix(".png").exists():
                found.add(path.with_suffix(".png"))
        for value in node.values():
            found.update(image_inputs(value))
    elif isinstance(node, list):
        for value in node:
            found.update(image_inputs(value))
    return found


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Source not found: {SOURCE}")
    text = SOURCE.read_text(encoding="utf-8")
    info = source_metadata(text)
    ast = json.loads(subprocess.check_output([
        "pandoc", str(SOURCE), "-f", "markdown+tex_math_dollars", "-t", "json",
    ]))
    # Unlike the paper, this plan has no abstract heading before section one.
    base.section_count = 1
    base.in_refs = False
    base.math_warnings.clear()
    story = base.convert(ast["blocks"])
    ExecutionPlanDoc(str(OUTPUT), info, sha256(SOURCE)).multiBuild(story)
    reader = base.PdfReader(OUTPUT)
    inputs = {SOURCE, Path(__file__).resolve(), RENDERER, *image_inputs(ast)}
    packages = {name: metadata.version(name)
                for name in ("reportlab", "matplotlib", "Pillow", "pypdf")}
    fonts = {path.name: sha256(path) for path in sorted(base.FONTDIR.glob("*.ttf"))}
    record = {
        "schema_version": 1,
        "document_id": info["document_id"],
        "version": info["version"],
        "document_status": info["status"],
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {repo_path(path): sha256(path) for path in sorted(inputs)},
        "outputs": {repo_path(OUTPUT): sha256(OUTPUT)},
        "pages": len(reader.pages),
        "build_environment": {
            "python": platform.python_version(),
            "packages": packages,
            "pandoc": subprocess.check_output(
                ["pandoc", "--version"], text=True).splitlines()[0],
            "font_sha256": fonts,
        },
        "math_width_warnings": base.math_warnings,
        "visual_review": {"status": "pending", "evidence": None},
        "scope": "CPU documentation build only; no experiment or approval is implied",
    }
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (CACHE / "extracted.txt").write_text(
        "\n\n".join(page.extract_text() or "" for page in reader.pages),
        encoding="utf-8")
    print(json.dumps({
        "document_id": info["document_id"], "version": info["version"],
        "pages": len(reader.pages), "pdf_bytes": OUTPUT.stat().st_size,
        "manifest": repo_path(MANIFEST),
        "math_width_warnings": base.math_warnings,
        "visual_review": "pending",
    }))


if __name__ == "__main__":
    main()
