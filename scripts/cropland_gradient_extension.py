"""
Declared extension (Methods: "Declared extension: the cropland gradient").

Pools NY-PA with IA, IL, IN, OH (2024-2025) and fits, per group at its NY-PA
radius (no re-selection):
  primary model + state fixed effects + crop x log_fsize + crop x edge_crop
Prediction: crop x log_fsize < 0, crop x edge_crop > 0.
Holm adjustment across the 6 interaction p-values. Conditional field-size and
edge odds ratios at the 25th/75th/90th percentiles of pooled cropland
fraction (site level). Robustness: no fields < 0.1 ha; top-10 observers removed.

  python cropland_gradient_extension.py NYPA_DIR CORNBELT_UPLOAD_DIR OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import reporting_rate_models as R

RADIUS = {"bumble_bee": 2000, "other_bee": 3000, "syrphid": 1000}
INTER = ["crop_x_fsize", "crop_x_edge"]
METRIC_COLS = ["cropland_frac", "edge_density_m_per_ha", "edge_per_crop_ha",
               "aw_mean_field_ha", "mean_field_ha"]


def avg_years(m24, m25):
    m = m25.merge(m24, on=["site_id", "radius_m"], suffixes=("", "_24"))
    for c in METRIC_COLS:
        m[c + "_avg"] = m[[c, c + "_24"]].mean(axis=1)
    return m


def build_pooled(nypa, cb, out):
    """Write a pooled data dir in the layout reporting_rate_models.load expects."""
    nypa, cb, out = Path(nypa), Path(cb), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cat = lambda a, b: pd.concat([a, b], ignore_index=True)
    s_n, s_c = pd.read_csv(nypa / "sites.csv"), pd.read_csv(cb / "sites_cb.csv")
    overlap = set(s_n.site_id) & set(s_c.site_id)
    assert not overlap, f"{len(overlap)} grid cells in both regions"
    cat(s_n, s_c).to_csv(out / "sites.csv", index=False)
    cat(pd.read_csv(nypa / "visits.csv"), pd.read_csv(cb / "visits_cb.csv")).to_csv(out / "visits.csv", index=False)
    cat(pd.read_csv(nypa / "detections.csv"), pd.read_csv(cb / "detections_cb.csv")).to_csv(out / "detections.csv", index=False)
    m24, m25 = pd.read_csv(cb / "site_metrics_cb_2024.csv"), pd.read_csv(cb / "site_metrics_cb_2025.csv")
    # data-quality exclusion: cropland > 100% means overlapping FTW polygons
    bad = set(m24.loc[m24.cropland_frac > 1, "site_id"]) | set(m25.loc[m25.cropland_frac > 1, "site_id"])
    print(f"excluded {len(bad)} sites with cropland fraction > 1 (overlapping polygons)")
    lcb = avg_years(m24[~m24.site_id.isin(bad)], m25[~m25.site_id.isin(bad)])
    cat(pd.read_csv(nypa / "landscape_avg.csv"), lcb).to_csv(out / "landscape_avg.csv", index=False)
    n24 = pd.read_csv(cb / "site_metrics_nosliver_cb_2024.csv")
    n25 = pd.read_csv(cb / "site_metrics_nosliver_cb_2025.csv")
    ncb = avg_years(n24[~n24.site_id.isin(bad)], n25[~n25.site_id.isin(bad)])
    cat(pd.read_csv(nypa / "landscape_avg_nosliver.csv"), ncb).to_csv(out / "landscape_avg_nosliver.csv", index=False)
    return out


def add_states(v):
    states = sorted(v.state.unique())
    base = "NY" if "NY" in states else states[0]
    dummies = []
    for st in states:
        if st == base:
            continue
        v[f"st_{st}"] = (v.state == st).astype(float); dummies.append(f"st_{st}")
    return v, dummies


def fit_one(v, land, guild, label):
    x = R.add_land(v, land, RADIUS[guild]).dropna(subset=R.LAND).reset_index(drop=True)
    x, st_terms = add_states(x)
    x["crop_x_fsize"] = x.crop * x.log_fsize
    x["crop_x_edge"] = x.crop * x.edge_crop
    ctrl = [c for c in R.CTRL if c != "pa"] + st_terms
    _, rob = R.fit(x, guild, R.LAND + INTER + ctrl)
    rows = []
    ci = np.exp(rob.conf_int())
    for t in R.LAND + INTER:
        rows.append({"guild": guild, "check": label, "kind": "coef", "term": t,
                     "visits": len(x), "sites": x.site_id.nunique(),
                     "odds_ratio": np.exp(rob.params[t]), "or_lo95": ci.loc[t, 0],
                     "or_hi95": ci.loc[t, 1], "p": rob.pvalues[t]})
    # conditional ORs at pooled site-level cropland percentiles
    site_crop = x.drop_duplicates("site_id").cropland_frac_avg
    mu, sd = x.cropland_frac_avg.mean(), x.cropland_frac_avg.std()
    V = rob.cov_params()
    for q in (0.25, 0.75, 0.90):
        cq = site_crop.quantile(q); z = (cq - mu) / sd
        for main, inter in (("log_fsize", "crop_x_fsize"), ("edge_crop", "crop_x_edge")):
            est = rob.params[main] + z * rob.params[inter]
            se = np.sqrt(V.loc[main, main] + z**2 * V.loc[inter, inter] + 2 * z * V.loc[main, inter])
            rows.append({"guild": guild, "check": label, "kind": "conditional",
                         "term": f"{main} @ crop p{int(q*100)} ({cq:.0%})",
                         "visits": len(x), "sites": x.site_id.nunique(),
                         "odds_ratio": np.exp(est), "or_lo95": np.exp(est - 1.96 * se),
                         "or_hi95": np.exp(est + 1.96 * se), "p": np.nan})
    return rows


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def run(pooled_dir, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    v, land = R.load(pooled_dir, [2024, 2025])
    land_ns = pd.read_csv(Path(pooled_dir) / "landscape_avg_nosliver.csv")
    top = v.observer.value_counts().head(10).index
    summary = v.groupby("state").agg(visits=("visit_id", "size"), sites=("site_id", "nunique"))
    print(summary.to_string())
    rows = []
    for g in RADIUS:
        rows += fit_one(v.copy(), land, g, "primary")
        rows += fit_one(v.copy(), land_ns, g, "no fields <0.1 ha")
        rows += fit_one(v[~v.observer.isin(top)].reset_index(drop=True), land, g, "top 10 observers removed")
    res = pd.DataFrame(rows)
    m = (res.check == "primary") & res.term.isin(INTER)
    res.loc[m, "p_holm"] = holm(res.loc[m, "p"])
    res.to_csv(out_dir / "extension_cropland_gradient.csv", index=False)
    summary.to_csv(out_dir / "extension_sample_by_state.csv")
    return res


if __name__ == "__main__":
    nypa, cb, out = sys.argv[1:4]
    pooled = build_pooled(nypa, cb, Path(out) / "data_pooled")
    r = run(pooled, out)
    show = r[(r.check == "primary")]
    print(show[["guild", "kind", "term", "odds_ratio", "or_lo95", "or_hi95", "p", "p_holm"]].round(3).to_string())
