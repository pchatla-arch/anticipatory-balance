#!/usr/bin/env python3
"""Build the candidate proposal without changing the parent review document.

Uses the existing documentation-only dependencies and Markdown/math helpers.
Reads proposal.md and analytical_check.json; writes proposal.pdf and two figures.
This builder does not run simulation, install dependencies, or regenerate checks.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("foresight_review_pdf", ROOT.parent / "build_paper.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load the parent documentation helpers")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

# Helper functions resolve these globals in their original module namespace.
# The parent source, figures, PDF, and cache are never used as output targets.
base.ROOT = ROOT
base.CACHE = ROOT / ".build"
base.CACHE.mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)
base.styles = copy.deepcopy(base.styles)
base.styles["title"].fontSize = 23
base.styles["title"].leading = 26.5
base.styles["title"].spaceAfter = 11
base.styles["body"].allowWidows = 0
base.styles["body"].allowOrphans = 0
base.math_warnings = []

TITLE = "Online Actuator Adaptation for Reliable Humanoid Locomotion"


def save_figure(fig, name: str) -> None:
    for suffix in ("svg", "png"):
        fig.savefig(ROOT / "figures" / f"{name}.{suffix}", dpi=240, facecolor="white")
    base.plt.close(fig)


def architecture_figure() -> None:
    """Draw the policy path and alternative inference paths separately."""
    plt = base.plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(10.5, 5.65))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.65)
    ax.axis("off")
    node, arrow = base.node, base.arrow

    node(ax, .20, 4.05, 2.25, .90, "Frozen PPO actor\nweights + normalization fixed", fc="#F4F4F4", fontsize=9.5)
    node(ax, 4.00, 4.05, 2.65, .90, "Bounded projection\nissued target + status", fc="#EAF3F7", fontsize=10)
    node(ax, 8.15, 4.05, 2.15, .90, "G1 actuator + plant\nhidden fault parameters", fc="#F4F4F4", fontsize=9.5)
    arrow(ax, [(2.45, 4.50), (4.00, 4.50)], "nominal target", (3.23, 4.73))
    arrow(ax, [(6.65, 4.50), (8.15, 4.50)], "effective command", (7.40, 4.73))

    node(ax, .20, 1.62, 2.25, 1.02, "Timestamped history\nstate, effort, issued targets\ncapture + arrival times", fontsize=9.4)
    node(ax, 3.38, 2.28, 2.05, .70, "Classical RLS\nphysical gain model", fontsize=9.2)
    node(ax, 3.38, 1.08, 2.05, .70, "Learned GRU\nmatched history baseline", fontsize=9.2)
    ax.text(4.405, 3.23, "ALTERNATIVE ESTIMATORS", ha="center", va="center", fontsize=8.6, color="#43565E", fontweight="bold")
    node(ax, 6.45, 1.55, 3.72, 1.38, "Gain set + state hold envelope\nmeasurement age and validity\ncommand / force authority", fc="#EAF3F7", fontsize=9.8)

    arrow(ax, [(2.45, 2.35), (2.85, 2.35), (2.85, 2.63), (3.38, 2.63)])
    arrow(ax, [(2.45, 1.93), (2.85, 1.93), (2.85, 1.43), (3.38, 1.43)])
    # The learned/classical estimate chooses the inverse candidate only.
    # The measurement-derived uncertainty set is shared and independent of it.
    arrow(ax, [(5.43, 2.63), (5.95, 2.63)])
    arrow(ax, [(5.43, 1.43), (5.95, 1.43), (5.95, 3.34), (4.75, 3.34), (4.75, 4.05)])
    ax.text(3.55, 3.65, "inverse candidate", ha="center", fontsize=8.3, color="#43565E")
    arrow(ax, [(2.45, 1.66), (2.75, 1.66), (2.75, .87), (7.20, .87), (7.20, 1.55)])
    ax.text(7.95, 1.10, "same physical evidence", ha="center", fontsize=8.1, color="#43565E")
    arrow(ax, [(8.30, 2.93), (8.30, 3.52), (5.32, 3.52), (5.32, 4.05)], "feasible correction set", (6.90, 3.73))
    arrow(ax, [(9.22, 4.05), (10.40, 4.05), (10.40, .65), (1.32, .65), (1.32, 1.62)])
    ax.text(7.00, .45, "telemetry arrives causally; simulator truth remains evaluation-only", ha="center", fontsize=8.2, color="#43565E")
    arrow(ax, [(9.22, 4.95), (9.22, 5.43), (1.32, 5.43), (1.32, 4.95)])
    ax.text(5.25, 5.52, "nominal actor observations", ha="center", fontsize=8.3, color="#43565E")
    ax.text(5.25, .16, "Proposed architecture. RLS and GRU share the same corrector; the gain oracle is a diagnostic arm.", ha="center", fontsize=8.4, color="#59646B")
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
    save_figure(fig, "research_architecture")


def authority_figure() -> None:
    """Render existing scalar calculations, never inferred robot measurements."""
    payload = json.loads((ROOT / "analytical_check.json").read_text())
    sweep = payload["sweep"]
    if not sweep or any(not {"upper_gain", "candidate", "effective"} <= row.keys() for row in sweep):
        raise ValueError("analytical_check.json lacks the expected scalar sweep")
    x = [row["upper_gain"] for row in sweep]
    candidate = [row["candidate"] for row in sweep]
    effective = [row["effective"] for row in sweep]
    plt = base.plt
    fig = plt.figure(figsize=(10.5, 4.45))
    ax = fig.add_axes([.085, .24, .625, .61])
    ax.axvspan(1.0, 1.05, color="#F0F2F3", zorder=0)
    ax.plot(x, candidate, color="#8C6065", linewidth=1.7, linestyle="--", label="Point-estimate inverse candidate")
    ax.plot(x, effective, color="#1B657C", linewidth=2.5, label="Projected effective correction")
    ax.fill_between(x, 0, effective, color="#DDEDF2", alpha=.7)
    ax.axvline(1.0, color="#7A858A", linewidth=1.0, linestyle=":")
    ax.set_xlim(.70, 1.05)
    ax.set_ylim(-.0015, .048)
    ax.set_xticks([.70, .75, .80, .85, .90, .95, 1.00, 1.05])
    ax.set_yticks([0, .01, .02, .03, .04])
    ax.set_xlabel("Upper end of plausible gain interval", fontsize=10.5, labelpad=7)
    ax.set_ylabel("Target correction (rad)", fontsize=10.5, labelpad=7)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color="#DDE2E5", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#819097")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, bbox_to_anchor=(1.0, 1.20))
    fig.text(.755, .80, "SAME POINT ESTIMATE", fontsize=10, fontweight="bold", color="#173747")
    fig.text(.755, .72, "Gain estimate: 0.70\nLower gain bound: 0.60\nTarget error: 0.08-0.12 rad\nControl hold: 20 ms\nSlew bound: 2 rad/s\nPrevious correction: 0 rad", fontsize=9.3, color="#43565E", va="top", linespacing=1.5)
    fig.text(.755, .35, "Wider uncertainty permits\nless correction here.\nIf gain 1 remains plausible,\nthis screen selects zero.", fontsize=9.3, color="#173747", va="top", linespacing=1.5)
    fig.text(.50, .065, "Illustrative scalar calculation; no robot measurements", ha="center", fontsize=10, fontweight="bold", color="#43565E")
    fig.text(.50, .015, "Source: analytical_check.json. Unsaturated model, bounded target and correction slew; no stability claim.", ha="center", fontsize=8.4, color="#59646B")
    save_figure(fig, "authority_example")


class ProposalDoc(base.PaperDoc):
    def __init__(self, path):
        base.BaseDocTemplate.__init__(
            self, path, pagesize=base.letter,
            leftMargin=42, rightMargin=42, topMargin=43, bottomMargin=42,
            title=TITLE, author="Prasuna Chatla",
            subject="Research proposal, analytical candidate and staged experimental protocol",
        )
        frame = base.Frame(42, 42, 528, 707, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(base.PageTemplate(id="body", frames=[frame], onPage=self.chrome))

    def chrome(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(base.MUTED)
        canvas.setFont("Body", 7.8)
        canvas.drawString(42, 768, "ONLINE ACTUATOR ADAPTATION  |  Research proposal")
        canvas.drawRightString(570, 768, "Selected plan 0.3.1")
        canvas.setStrokeColor(base.LINE)
        canvas.setLineWidth(.5)
        canvas.line(42, 759, 570, 759)
        canvas.line(42, 32, 570, 32)
        canvas.setFont("Body", 7.8)
        canvas.drawString(42, 20, "Prasuna Chatla  |  November 15, 2026")
        canvas.drawRightString(570, 20, str(doc.page))
        canvas.restoreState()


def main() -> None:
    architecture_figure()
    authority_figure()
    source = ROOT / "proposal.md"
    if not source.exists():
        raise SystemExit("Figures created. proposal.md is not present yet; run again after it is written.")
    ast = json.loads(subprocess.check_output([
        "pandoc", str(source), "-f", "markdown+tex_math_dollars", "-t", "json",
    ]))
    base.section_count = 0
    base.in_refs = False
    base.math_warnings.clear()
    story = base.convert(ast["blocks"])
    # Keep this short equation introduction with its displayed optimization.
    for index, item in enumerate(story):
        if (isinstance(item, base.Paragraph)
                and item.getPlainText() == "The single primary optimization is"):
            equation = story[index + 1]
            content = (equation._content if isinstance(equation, base.KeepTogether)
                       else [equation])
            story[index:index + 2] = [base.KeepTogether([item, *content])]
            break
    pdf = ROOT / "proposal.pdf"
    ProposalDoc(str(pdf)).multiBuild(story)
    reader = base.PdfReader(pdf)
    stats = {
        "pages": len(reader.pages), "bytes": pdf.stat().st_size,
        "words": len(source.read_text().split()),
        "math_width_warnings": base.math_warnings,
        "analytical_figure_source": "analytical_check.json",
        "scope": "Documentation build only; no robot experiments",
    }
    (base.CACHE / "build_report.json").write_text(json.dumps(stats, indent=2) + "\n")
    (base.CACHE / "extracted.txt").write_text("\n\n".join(page.extract_text() or "" for page in reader.pages))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
