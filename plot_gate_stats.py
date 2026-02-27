import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _pivot_heat(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    piv = df.pivot_table(
        index="step_index",
        columns="block_index",
        values=value_col,
        aggfunc="mean",
    ).sort_index()
    piv = piv.reindex(sorted(piv.columns), axis=1)
    return piv


def _save_heatmap(piv: pd.DataFrame, title: str, outpath: str):
    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = plt.gca()
    im = ax.imshow(
        piv.values,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )
    ax.set_title(title)
    ax.set_xlabel("block_index")
    ax.set_ylabel("step_index")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(int(c)) for c in piv.columns], rotation=90)
    ax.set_yticks(np.linspace(0, len(piv.index) - 1, num=min(10, len(piv.index)), dtype=int))
    ax.set_yticklabels([str(int(piv.index[i])) for i in ax.get_yticks()])

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def _save_topblocks_lines(piv: pd.DataFrame, title: str, outpath: str, top_k: int = 5):
    # choose blocks with largest max across steps
    col_max = piv.max(axis=0).sort_values(ascending=False)
    top_blocks = list(col_max.head(top_k).index)

    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax = plt.gca()
    for b in top_blocks:
        ax.plot(piv.index.values, piv[b].values, label=f"block {int(b)}")
    ax.set_title(title)
    ax.set_xlabel("step_index")
    ax.set_ylabel(piv.columns.name or "value")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to gate_stats.csv")
    ap.add_argument("--outdir", type=str, default="plots", help="Output directory")
    ap.add_argument("--top_k", type=int, default=5, help="How many blocks to plot as lines")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    # basic hygiene
    for col in ["kind", "step_index", "block_index"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV.")

    df["step_index"] = df["step_index"].astype(int)
    df["block_index"] = df["block_index"].astype(int)

    for kind in ["dual", "single"]:
        d = df[df["kind"] == kind].copy()
        if len(d) == 0:
            print(f"[warn] no rows for kind='{kind}'")
            continue

        # mean_g heatmap + top-block lines
        if "mean_g" in d.columns:
            piv_g = _pivot_heat(d, "mean_g")
            _save_heatmap(
                piv_g,
                title=f"mean_g heatmap ({kind})",
                outpath=os.path.join(args.outdir, f"mean_g_heatmap_{kind}.png"),
            )
            _save_topblocks_lines(
                piv_g,
                title=f"mean_g vs step (top {args.top_k} blocks) ({kind})",
                outpath=os.path.join(args.outdir, f"mean_g_vs_step_topblocks_{kind}.png"),
                top_k=args.top_k,
            )
        else:
            print(f"[warn] missing 'mean_g' for kind='{kind}'")

        # mean_r heatmap
        if "mean_r" in d.columns:
            piv_r = _pivot_heat(d, "mean_r")
            _save_heatmap(
                piv_r,
                title=f"mean_r heatmap ({kind})",
                outpath=os.path.join(args.outdir, f"mean_r_heatmap_{kind}.png"),
            )
        else:
            print(f"[warn] missing 'mean_r' for kind='{kind}'")

    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()