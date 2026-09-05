"""Render results.json files into the markdown tables used in the writeup.

Kept separate from evaluation so that tables are regenerated from committed
artifacts rather than by re-running experiments, which is what makes the
reported numbers reproducible from the repository alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_pair(payload: dict) -> str:
    """One pair's results as a markdown section."""
    source = payload["source"].split("/")[-1]
    target = payload["target"].split("/")[-1]
    lines = [
        f"### {source} to {target}",
        "",
        f"Calibrated on {payload['n_fit_tokens']:,} tokens "
        f"({payload['n_val_tokens']:,} held out), "
        f"k={payload['settings']['k']} source layers per target layer.",
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

    return "\n".join(lines)


def summarize_r2(payload: dict) -> str:
    """Compare held-out fit quality across variants.

    Included because the reference work's central diagnostic is that this
    quantity fails to predict retention. Reporting both side by side is what
    lets a reader check that claim against these runs.
    """
    lines = ["### Held-out fit quality vs. retention", "", "| variant | mean held-out R² (keys) | mean held-out R² (values) |", "| --- | --- | --- |"]
    for variant, diag in payload.get("diagnostics", {}).items():
        keys = [d["held_out_r2"] for d in diag["keys"].values()]
        values = [d["held_out_r2"] for d in diag["values"].values()]
        lines.append(
            f"| {variant} | {sum(keys) / len(keys):.4f} | {sum(values) / len(values):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="directory to scan")
    parser.add_argument("--out", default="results/TABLES.md")
    args = parser.parse_args()

    files = sorted(Path(args.results).rglob("results.json"))
    if not files:
        raise SystemExit(f"no results.json under {args.results}")

    sections = ["# Results", ""]
    for path in files:
        payload = json.loads(path.read_text())
        sections.append(format_pair(payload))
        sections.append(summarize_r2(payload))

    Path(args.out).write_text("\n".join(sections))
    print(f"wrote {args.out} from {len(files)} result file(s)")


if __name__ == "__main__":
    main()
