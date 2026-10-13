"""Figures for the manuscript (Frontiers: TIFF, 300 DPI, 8.5-18 cm wide)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6})
TERMS = [("log_fsize", "Field size\n(log area-weighted)"),
         ("edge_crop", "Edge per ha\nof cropland"),
         ("crop", "Cropland\nfraction")]
PER = [("2024_2025", "2024–25 (primary)", "#1b6ca8", "o"),
       ("2017_2018_2019_2020_2021_2022_2023", "2017–23 (replication)", "#d98c2b", "s")]
BEST = {"bumble_bee": 2000, "other_bee": 1000, "syrphid": 1000}
GNAME = {"bumble_bee": "Bumble bees", "other_bee": "Other bees", "syrphid": "Hoverflies"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)


def forest(ax_row, data, labels, title_row=True):
    for j, (t, tlabel) in enumerate(TERMS):
        ax = ax_row[j]
        for k, (tag, plab, col, mk) in enumerate(PER):
            d = data[tag]
            for i, lab in enumerate(labels):
                r = d[(d.label == lab) & (d.term == t)]
                if r.empty: continue
                y = i + (k - 0.5) * 0.3
                ax.errorbar(r.odds_ratio, y, xerr=[[r.odds_ratio.iat[0] - r.or_lo95.iat[0]],
                            [r.or_hi95.iat[0] - r.odds_ratio.iat[0]]], fmt=mk, color=col,
                            ms=3.5, lw=0.9, capsize=0, label=plab if (i == 0 and j == 0) else None)
        ax.axvline(1, color="0.5", lw=0.6, ls="--")
        ax.set_xscale("log"); ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.minorticks_off(); ax.set_xlim(0.38, 3.2)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if j == 0 else [])
        ax.invert_yaxis()
        if title_row: ax.set_title(tlabel, fontsize=8)
        ax.set_xlabel("Odds ratio (per SD)")
        for s in ["top", "right"]: ax.spines[s].set_visible(False)


# Figure 2: groups, best radius, both periods
data = {}
for tag, *_ in PER:
    c = pd.read_csv(RES / f"rr_coefficients_{tag}.csv")
    c = c[c.apply(lambda r: BEST[r.guild] == r.radius_m, axis=1)].copy()
    c["label"] = c.guild.map(GNAME) + "\n(" + (c.radius_m // 1000).astype(str) + " km)"
    data[tag] = c
labels = [f"{GNAME[g]}\n({r // 1000} km)" for g, r in BEST.items()]
fig, axes = plt.subplots(1, 3, figsize=(17 * CM, 6 * CM))
forest(axes, data, labels)
fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "Figure2_group_effects")

# Figure 3: declared Bombus species analysis
sp_tags = {"2024_2025": "secondary_bombus_species_2024_2025.csv",
           "2017_2018_2019_2020_2021_2022_2023": "secondary_bombus_species_2017_2023.csv"}
data = {}
for tag, f in sp_tags.items():
    c = pd.read_csv(RES / f)
    c["label"] = c.response.str.replace("Bombus ", "B. ", regex=False)
    data[tag] = c
labels = list(data["2024_2025"].label.unique())
fig, axes = plt.subplots(1, 3, figsize=(17 * CM, 9.5 * CM))
forest(axes, data, labels)
for lab in axes[0].get_yticklabels():
    if lab.get_text().startswith("B. ") and "excl." not in lab.get_text(): lab.set_fontstyle("italic")
fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, "Figure3_bombus_species")

# Figure S1: radius selection
fig, axes = plt.subplots(1, 2, figsize=(14 * CM, 5.5 * CM), sharey=True)
for ax, (tag, plab, *_ ) in zip(axes, PER):
    s = pd.read_csv(RES / f"rr_model_selection_{tag}.csv")
    for g, col in zip(GNAME, ["#1b6ca8", "#5aa469", "#c0504d"]):
        d = s[s.guild == g]
        ax.plot(d.radius_m / 1000, d.dAIC, "-o", color=col, ms=3.5, lw=1, label=GNAME[g])
    ax.set_title(plab, fontsize=8); ax.set_xlabel("Radius (km)"); ax.set_xticks([1, 2, 3])
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
axes[0].set_ylabel("ΔAIC (vs best radius)"); axes[0].legend(frameon=False)
fig.tight_layout(); save(fig, "FigureS1_radius_selection")
print("ok")
