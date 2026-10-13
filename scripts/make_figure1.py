"""Figure 1: study region, all survey cells, and agricultural cells analyzed."""
import sys; sys.path.insert(0, ".")
import geopandas as gpd, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import reporting_rate_models as R
CM = 1 / 2.54
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
st = gpd.read_file(sys.argv[1]).to_crs(5070)
s = pd.read_csv("data_v3/sites.csv")
v24, _ = R.load("data_v3", [2024, 2025])
v17, _ = R.load("data_v3", list(range(2017, 2024)))
g = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.lon, s.lat), crs=4326).to_crs(5070)
g = g[g.within(st.union_all())]
a24 = g.site_id.isin(v24.site_id); a17 = g.site_id.isin(v17.site_id) & ~a24
fig, ax = plt.subplots(figsize=(17 * CM, 10 * CM))
st.boundary.plot(ax=ax, color="0.2", lw=0.6)
g[~(a24 | a17)].plot(ax=ax, color="0.78", markersize=0.4, label="Survey cell, <5% cropland (excluded)")
g[a17].plot(ax=ax, color="#d98c2b", markersize=1.2, label="Agricultural cell, 2017–23 only")
g[a24].plot(ax=ax, color="#1b6ca8", markersize=1.2, label="Agricultural cell, 2024–25 analysis")
for _, r in st.iterrows():
    c = r.geometry.representative_point(); ax.text(c.x, c.y, r.STUSPS, fontsize=11, weight="bold", color="0.25", ha="center")
ax.set_axis_off()
from matplotlib.lines import Line2D
h = [Line2D([], [], ls="", marker="o", ms=4, color=c) for c in ["0.7", "#d98c2b", "#1b6ca8"]]
fig.legend(h, ["Survey cell, <5% cropland (excluded)", "Agricultural cell, 2017–23 only",
              "Agricultural cell, 2024–25 (primary)"], loc="lower center", ncol=3, frameon=False, fontsize=7)
x0 = st.total_bounds[0] + 20000; y0 = st.total_bounds[1] + 15000
ax.plot([x0, x0 + 100000], [y0, y0], color="k", lw=1.5); ax.text(x0 + 50000, y0 + 8000, "100 km", ha="center")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("figures/Figure1_study_region.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig("figures/Figure1_study_region.png", dpi=150)
print(len(g), a24.sum(), a17.sum())
