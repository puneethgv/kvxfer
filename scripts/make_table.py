"""Render evaluation result files into the markdown tables used in the writeup.

Kept separate from evaluation so that tables are regenerated from committed
artifacts rather than by re-running experiments, which is what makes the
reported numbers reproducible from the repository alone.

Every ``results*.json`` under the scanned directory is rendered, not only
``results.json``. A pair evaluated a second time -- to add the trained residual,
say -- writes ``results_residual.json`` alongside the first run, and globbing
the exact name silently dropped it: the residual is the only variant here that
improved on plain ridge and it was missing from the generated tables entirely.
The suffix becomes the run label so the two sections stay distinguishable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_perplexity(payload: dict) -> str:
    """The primary metric, with the paired tests that make it decisive.

    Perplexity leads because it yields a measurement per token rather than per
    item. On these pairs the multiple-choice tasks leave so little headroom
    between the mapped cache and the target's own prefill that per-item noise
    swamps the effect being measured; per-document negative log likelihood does
    not have that problem.
    """
    ppl = payload.get("perplexity")
    if not ppl:
        return ""

    lines = [
        "**Prefix-conditioned perplexity** (primary metric)",
        "",
        "| condition | perplexity | mean NLL | stored params |",
        "| --- | --- | --- | --- |",
    ]
    diagnostics = payload.get("diagnostics", {})
    for name, result in ppl.items():
        stored = diagnostics.get(name, {}).get("stored_parameters")
        size = "—" if stored is None else f"{stored / 1e6:.1f}M"
        lines.append(
            f"| {name} | {result['perplexity']:.4f} | "
            f"{result['mean_nll']:.5f} ± {result['stderr']:.5f} | {size} |"
        )
    lines.append("")

    paired = payload.get("perplexity_paired") or {}
    if paired:
        lines += [
            "Paired by document, so the comparison is not read off "
            "overlapping per-condition error bars:",
            "",
            "| comparison | mean NLL difference | t | documents improved |",
            "| --- | --- | --- | --- |",
        ]
        for label, test in paired.items():
            lines.append(
                f"| {label.replace('_vs_', ' vs ')} | "
                f"{test['mean_difference']:+.5f} ± {test['stderr']:.5f} | "
                f"{test['t_statistic']:+.2f} | "
                f"{test['n_better']}/{test['n_documents']} |"
            )
        lines.append("")
    return "\n".join(lines)


def run_label(path: Path) -> str:
    """The part of a result filename that distinguishes one run from another.

    ``results.json`` is the unlabelled run; ``results_residual.json`` is the
    "residual" run of the same pair. Without this the two render as sections
    with identical headings.
    """
    return path.stem.removeprefix("results").lstrip("_").replace("_", " ")


def format_pair(payload: dict, label: str = "") -> str:
    """One pair's results as a markdown section.

    Args:
        payload: a parsed result file.
        label: run label from :func:`run_label`, appended to the heading when
            a pair has more than one result file.
    """
    source = payload["source"].split("/")[-1]
    target = payload["target"].split("/")[-1]
    heading = f"### {source} to {target}"
    if label:
        heading += f" ({label} run)"

    conditions = sorted(
        {name for block in payload["results"].values() for name in block["conditions"]}
        | set(payload.get("perplexity") or {})
    )
    lines = [
        heading,
        "",
        f"Calibrated on {payload['n_fit_tokens']:,} tokens "
        f"({payload['n_val_tokens']:,} held out), "
        f"k={payload['settings']['k']} source layers per target layer.",
        "",
        f"Conditions: {', '.join(conditions)}.",
        "",
    ]

    for task, block in payload["results"].items():
        conditions = block["conditions"]
        lines += [
            f"**{task}** ({conditions['target']['n_items']} items)",
            "",
            "| condition | accuracy | retention | floor-normalized |",
            "| --- | --- | --- | --- |",
        ]
        for name, cond in conditions.items():
            accuracy = f"{cond['accuracy']:.3f} ± {cond['stderr']:.3f}"
            if name == "target":
                retention, normalized = "—", "—"
            else:
                retention = f"{block['retention'].get(name, float('nan')):.1%}"
                value = block["floor_normalized_retention"].get(name)
                normalized = "—" if value is None else f"{value:.1%}"
            lines.append(f"| {name} | {accuracy} | {retention} | {normalized} |")
        lines.append("")

        for label, test in (block.get("paired") or {}).items():
            lines.append(
                f"Paired {label.replace('_vs_', ' vs ')}: "
                f"{test['difference']:+.4f} accuracy, exact McNemar "
                f"p={test['p_value']:.3f}."
            )
        if block.get("paired"):
            lines.append("")

    return "\n".join(lines)


def summarize_r2(payload: dict) -> str:
    """Compare held-out fit quality across variants.

    Included because the reference work's central diagnostic is that this
    quantity fails to predict retention. Reporting both side by side is what
    lets a reader check that claim against these runs.
    """
    rows = []
    for variant, diag in payload.get("diagnostics", {}).items():
        keys = [d["held_out_r2"] for d in diag.get("keys", {}).values()]
        values = [d["held_out_r2"] for d in diag.get("values", {}).values()]
        if not keys or not values:
            # A trained variant has no closed-form diagnostics to report.
            continue
        rows.append(
            f"| {variant} | {sum(keys) / len(keys):.4f} | {sum(values) / len(values):.4f} |"
        )
    if not rows:
        return ""

    lines = [
        "### Held-out fit quality vs. retention",
        "",
        "| variant | mean held-out R² (keys) | mean held-out R² (values) |",
        "| --- | --- | --- |",
        *rows,
    ]
    lines += [
        "",
        "Read this against the perplexity table rather than on its own. The "
        "reference work reports calibration R² anti-correlating with retention "
        "(r = -0.20), so a variant leading here is not thereby the better "
        "mapper -- that is the claim these runs are set up to check.",
    ]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="directory to scan")
    parser.add_argument("--out", default="results/TABLES.md")
    args = parser.parse_args()

    files = sorted(Path(args.results).rglob("results*.json"))
    if not files:
        raise SystemExit(f"no results*.json under {args.results}")

    # Only label a run when its pair has more than one result file, so the
    # common single-run case keeps a clean heading.
    per_directory: dict[Path, int] = {}
    for path in files:
        per_directory[path.parent] = per_directory.get(path.parent, 0) + 1

    sections = ["# Results", ""]
    for path in files:
        payload = json.loads(path.read_text())
        label = run_label(path) if per_directory[path.parent] > 1 else ""
        sections.append(format_pair(payload, label))
        sections.append(format_perplexity(payload))
        sections.append(summarize_r2(payload))

    Path(args.out).write_text("\n".join(sections))
    print(f"wrote {args.out} from {len(files)} result file(s)")


if __name__ == "__main__":
    main()
