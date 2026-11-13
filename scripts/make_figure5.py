"""Figure 5: edge per cropland ha, all visits vs visits with all records located to <=100 m."""
import sys; from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
d = pd.read_csv(Path(sys.argv[1]) / "accuracy_subsets.csv"); out = Path(sys.argv[2])
d = d[(d.term == "edge_crop") & d.check.str.contains("all visits|all records")].copy()
d["region"] = d.check.str.split(" \\| ").str[0]; d["subset"] = d.check.str.split(" \\| ").str[1]
CM = 1 / 2.54; plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
G = {"bumble_bee": "Bumble bees (2 km)", "other_bee": "Other bees (3 km)", "syrphid": "Hoverflies (1 km)"}
REG = ["NY-PA", "Corn Belt", "Six states"]; COL = {"all visits": "#9e9e9e", "all records <=100 m": "#1b6ca8"}
fig, axes = plt.subplots(1, 3, figsize=(18 * CM, 6.5 * CM))
for ax, (g, title) in zip(axes, G.items()):
    for i, reg in enumerate(REG):
        for k, sub in enumerate(COL):
            r = d[(d.guild == g) & (d.region == reg) & (d.subset == sub)].iloc[0]
            y = i + (k - 0.5) * 0.3
            ax.errorbar(r.odds_ratio, y, xerr=[[r.odds_ratio - r.or_lo95], [r.or_hi95 - r.odds_ratio]],
                        fmt="o", color=COL[sub], ms=3.5, lw=1, capsize=0)
    ax.axvline(1, color="0.5", ls="--", lw=0.6); ax.set_xscale("log"); ax.set_xticks([0.75, 1, 1.25, 1.5])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g")); ax.minorticks_off(); ax.set_xlim(0.7, 1.6)
    ax.set_yticks(range(3)); ax.set_yticklabels(REG if ax is axes[0] else []); ax.invert_yaxis()
    ax.set_title(title, fontsize=8); ax.set_xlabel("Odds ratio (per SD)")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
h = [Line2D([], [], ls="", marker="o", color=c) for c in COL.values()]
fig.legend(h, ["All visits", "Visits with all records located to ≤100 m"], loc="lower center", ncol=2, frameon=False)
fig.suptitle("Edge per hectare of cropland", fontsize=9); fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(out / "Figure5_positional_accuracy.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(out / "Figure5_positional_accuracy.png", dpi=150); print("ok")
