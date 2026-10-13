"""Figure 1 (six-state study region) and Figure 4 (declared cropland-gradient extension)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).parent)); import reporting_rate_models as R

POOLED, EXT, STATES_GEOJSON, OUT = map(Path, sys.argv[1:5]); OUT.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
def save(fig, name):
    fig.savefig(OUT / f"{name}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{name}.png", dpi=150); plt.close(fig)

# ---- Figure 1 -------------------------------------------------------------
NAMES = {"New York": "NY", "Pennsylvania": "PA", "Ohio": "OH", "Indiana": "IN", "Illinois": "IL", "Iowa": "IA"}
st = gpd.read_file(STATES_GEOJSON); st = st[st.name.isin(NAMES)].to_crs(5070)
s = pd.read_csv(POOLED / "sites.csv")
v, land = R.load(POOLED, [2024, 2025])
crop = land[land.radius_m == 2000].set_index("site_id").cropland_frac_avg
g = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.lon, s.lat), crs=4326).to_crs(5070)
ana = g.site_id.isin(v.site_id)
fig, ax = plt.subplots(figsize=(18 * CM, 8.5 * CM))
st.plot(ax=ax, color="white", edgecolor="0.25", lw=0.6)
g[~ana].plot(ax=ax, color="0.8", markersize=0.25)
ga = g[ana].assign(crop=lambda d: d.site_id.map(crop)).sort_values("crop")
sc = ax.scatter(ga.geometry.x, ga.geometry.y, c=ga.crop, cmap="viridis", vmin=0, vmax=0.8, s=1.6, lw=0)
for _, r in st.iterrows():
    c = r.geometry.representative_point()
    ax.text(c.x, c.y, NAMES[r["name"]], fontsize=9, weight="bold", color="0.15", ha="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
x0, y0 = st.total_bounds[0] + 30000, st.total_bounds[1] + 20000
ax.plot([x0, x0 + 200000], [y0, y0], color="k", lw=1.5); ax.text(x0 + 100000, y0 + 15000, "200 km", ha="center")
ax.set_axis_off()
cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01); cb.set_label("Cropland fraction within 2 km")
fig.legend([Line2D([], [], ls="", marker="o", ms=3, color="0.75")], ["Survey cell not analyzed"],
           loc="lower left", frameon=False, bbox_to_anchor=(0.08, 0.02))
fig.tight_layout(); save(fig, "Figure1_study_region")

# ---- Figure 4 -------------------------------------------------------------
r = pd.read_csv(EXT / "extension_cropland_gradient.csv")
r = r[(r.check == "primary") & (r.kind == "conditional")].copy()
r["metric"] = np.where(r.term.str.startswith("log_fsize"), "Field size (log area-weighted)", "Edge per ha of cropland")
r["pct"] = r.term.str.extract(r"p(\d+)").astype(int)
r["cropv"] = r.term.str.extract(r"\((\d+)%\)").astype(int)
GN = {"bumble_bee": "Bumble bees (2 km)", "other_bee": "Other bees (3 km)", "syrphid": "Hoverflies (1 km)"}
COL = {25: "#9ecae1", 75: "#3182bd", 90: "#08306b"}
fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 6.5 * CM))
for ax, metric in zip(axes, ["Field size (log area-weighted)", "Edge per ha of cropland"]):
    for i, gname in enumerate(GN):
        d = r[(r.metric == metric) & (r.guild == gname)].sort_values("pct")
        for k, (_, row) in enumerate(d.iterrows()):
            y = i + (k - 1) * 0.25
            ax.errorbar(row.odds_ratio, y, xerr=[[row.odds_ratio - row.or_lo95], [row.or_hi95 - row.odds_ratio]],
                        fmt="o", color=COL[row.pct], ms=3.5, lw=1, capsize=0)
    ax.axvline(1, color="0.5", ls="--", lw=0.6); ax.set_xscale("log")
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.minorticks_off(); ax.set_xlim(0.5, 2.0)
    ax.set_yticks(range(len(GN))); ax.set_yticklabels(list(GN.values()) if ax is axes[0] else [])
    ax.invert_yaxis(); ax.set_title(metric, fontsize=8); ax.set_xlabel("Odds ratio (per SD)")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
h = [Line2D([], [], ls="", marker="o", color=COL[p]) for p in (25, 75, 90)]
fig.legend(h, ["25th percentile of cropland", "75th percentile", "90th percentile"], loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "Figure4_cropland_gradient")
print("ok")
