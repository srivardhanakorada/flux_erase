import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTDIR = Path("coip_analysis_plots")
OUTDIR.mkdir(exist_ok=True)

BASIS_JSON = "coip_analysis_outputs/basis_overlap_coip.json"
GATE_JSON = "coip_analysis_outputs/gate_stats_coip.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_basis_heatmap(basis_data, stream, metric):
    block_dict = basis_data.get(stream, {})
    blocks = sorted(int(k) for k in block_dict.keys())
    vals = [block_dict[str(b)][metric] if str(b) in block_dict else block_dict[b][metric] for b in blocks]

    arr = np.array(vals, dtype=float)[None, :]

    plt.figure(figsize=(12, 2.2))
    plt.imshow(arr, aspect="auto",vmin=0.0, vmax=0.25)
    plt.yticks([0], [metric])
    plt.xticks(range(len(blocks)), blocks)
    plt.xlabel("Block index")
    plt.title(f"{stream.capitalize()} blocks: {metric}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{stream}_{metric}.png", dpi=200)
    plt.close()


def flatten_gate_logs(gate_data, stream):
    rows = []
    stream_dict = gate_data.get(stream, {})
    for blk_key, entries in stream_dict.items():
        blk = int(blk_key)
        for e in entries:
            row = {"block": blk}
            row.update(e)
            rows.append(row)
    return rows


def grouped_metric(rows, metric):
    groups = {}
    for r in rows:
        tag = r.get("prompt_tag", "unknown")
        val = r.get(metric, None)
        if val is None:
            continue
        groups.setdefault(tag, []).append(float(val))
    return groups


def plot_group_box(rows, metric, stream):
    groups = grouped_metric(rows, metric)
    labels = list(groups.keys())
    data = [groups[k] for k in labels]

    if not data:
        return

    plt.figure(figsize=(8, 4.5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel(metric)
    plt.title(f"{stream.capitalize()} blocks: {metric} by prompt group")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{stream}_{metric}_box.png", dpi=200)
    plt.close()


def plot_blockwise_group_means(rows, metric, stream):
    tags = sorted(set(r.get("prompt_tag", "unknown") for r in rows))
    blocks = sorted(set(int(r["block"]) for r in rows))

    plt.figure(figsize=(11, 5))
    for tag in tags:
        ys = []
        for blk in blocks:
            vals = [float(r[metric]) for r in rows if r.get("prompt_tag", "unknown") == tag and int(r["block"]) == blk and metric in r]
            ys.append(np.mean(vals) if vals else np.nan)
        plt.plot(blocks, ys, marker="o", label=tag)

    plt.xlabel("Block index")
    plt.ylabel(metric)
    plt.title(f"{stream.capitalize()} blocks: mean {metric} by block")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{stream}_{metric}_blockwise.png", dpi=200)
    plt.close()


def main():
    basis_data = load_json(BASIS_JSON)
    gate_data = load_json(GATE_JSON)

    # Basis overlap plots
    for stream in ["dual", "single"]:
        for metric in ["U_vs_Vperson", "U_vs_Vret"]:
            plot_basis_heatmap(basis_data, stream, metric)

    # Gate plots
    for stream in ["dual", "single"]:
        rows = flatten_gate_logs(gate_data, stream)
        for metric in ["rU_mean", "rP_mean", "margin_mean", "gate_mean"]:
            plot_group_box(rows, metric, stream)
            plot_blockwise_group_means(rows, metric, stream)

    print(f"Saved plots to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()