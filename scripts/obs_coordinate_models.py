"""
Reviewer 2, point 2: landscape metrics at each record's own coordinates
(records with known positional accuracy <= 100 m), averaged within visits.
Point rule mirrors the cell rule: >= 5% FTW cropland at every radius
(cropland capped at 1). Fits NY-PA only, Corn Belt only (independent of the
region where the bumble bee edge signal was first seen), all six states, and
the declared extension model with interactions.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import reporting_rate_models as R
import reviewer_sensitivity as S

D = Path("data_obs")
MET = ["cropland_frac", "aw_mean_field_ha", "edge_per_crop_ha"]


def point_metrics(prefix):
    a = pd.read_csv(D / f"{prefix}_2025.csv").merge(pd.read_csv(D / f"{prefix}_2024.csv"),
                                                    on=["site_id", "radius_m"], suffixes=("", "_24"))
    for c in MET:
        if c == "cropland_frac":
            a[c] = a[c].clip(upper=1); a[c + "_24"] = a[c + "_24"].clip(upper=1)
        a[c + "_avg"] = a[[c, c + "_24"]].mean(axis=1)
    ok = (a[a.cropland_frac_avg >= 0.05].dropna(subset=[c + "_avg" for c in MET])
          .groupby("site_id").radius_m.nunique())
    return a[a.site_id.isin(ok[ok == 3].index)].rename(columns={"site_id": "pid"})


def records(path):
    r = pd.read_csv(path); r = r[r.pos_acc_m.notna() & (r.pos_acc_m <= 100)].copy()
    r["pid"] = r.x.round(0).astype(int).astype(str) + "_" + r.y.round(0).astype(int).astype(str)
    return r[["visit_id", "pid"]]


def visit_land(rec, pm, radius):
    p = pm[pm.radius_m == radius]
    return rec.merge(p, on="pid").groupby("visit_id")[[c + "_avg" for c in MET]].mean()


def build(v, vl, winsor=False):
    x = v[v.visit_id.isin(vl.index)].copy().reset_index(drop=True)
    for c in vl.columns: x[c] = x.visit_id.map(vl[c])
    e = x.edge_per_crop_ha_avg
    if winsor: e = e.clip(upper=e.quantile(0.99))
    x["crop"] = R.zs(x.cropland_frac_avg); x["log_fsize"] = R.zs(np.log(x.aw_mean_field_ha_avg)); x["edge_crop"] = R.zs(e)
    return x


def terms_for(x, interactions):
    t = list(R.LAND) + [c for c in R.CTRL if c != "pa"]
    t += [c for c in x.columns if c.startswith("yr") and c[2:].isdigit()]
    states = sorted(x.state.unique())
    for st in states[1:]:
        x[f"st_{st}"] = (x.state == st).astype(float); t.append(f"st_{st}")
    if interactions:
        x["crop_x_fsize"] = x.crop * x.log_fsize; x["crop_x_edge"] = x.crop * x.edge_crop; t += S.INTER
    return t


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    v, _ = R.load("results_ext/data_pooled", [2024, 2025])
    rec = pd.concat([records(D / "records_nypa.csv"), records(D / "records_cb.csv")])
    pm = pd.concat([point_metrics("obs_metrics_nypa"), point_metrics("obs_metrics_cb")])
    rows = []
    for g, r in S.EXT_R.items():
        vl = visit_land(rec, pm, r)
        for winsor in (False, True):
            tag = " (edge capped at p99)" if winsor else ""
            for region, mask in [("NY-PA", v.state.isin(["NY", "PA"])), ("Corn Belt", ~v.state.isin(["NY", "PA"])), ("six states", v.state.notna())]:
                x = build(v[mask], vl, winsor); t = terms_for(x, False)
                rob, _ = S.fit(x, g, t, pd.factorize(x.site_id)[0])
                rows += S.rows_from(rob, ["log_fsize", "edge_crop", "crop"], g, f"obs coords, {region}{tag}", len(x), "obs-coordinate")
            x = build(v, vl, winsor); t = terms_for(x, True)
            rob, _ = S.fit(x, g, t, pd.factorize(x.site_id)[0])
            rows += S.rows_from(rob, S.INTER, g, f"obs coords, declared extension model{tag}", len(x), "obs-coordinate")
    d = pd.DataFrame(rows); d.to_csv(out / "obs_coordinate_models.csv", index=False)
    f = lambda a: f"{a.odds_ratio:.2f} ({a.or_lo95:.2f}-{a.or_hi95:.2f}) p={a.p:.3f}"
    for (g, ck), dd in d.groupby(["guild", "check"], sort=False):
        print(g, "|", ck, "| n=", dd.visits.iloc[0], "|", " ; ".join(f"{a.term}: {f(a)}" for a in dd.itertuples()))


def matched_comparison(out):
    """Cell-centre vs record-coordinate metrics on the identical visits, per region."""
    v, land = R.load("results_ext/data_pooled", [2024, 2025])
    rec = pd.concat([records(D / "records_nypa.csv"), records(D / "records_cb.csv")])
    pm = pd.concat([point_metrics("obs_metrics_nypa"), point_metrics("obs_metrics_cb")])
    rows = []
    for g, r in S.EXT_R.items():
        vl = visit_land(rec, pm, r)
        for region, mask in [("NY-PA", v.state.isin(["NY", "PA"])), ("Corn Belt", ~v.state.isin(["NY", "PA"])), ("Six states", v.state.notna())]:
            xo = build(v[mask], vl); t = terms_for(xo, False)
            rob, _ = S.fit(xo, g, t, pd.factorize(xo.site_id)[0])
            rows += S.rows_from(rob, ["log_fsize", "edge_crop", "crop"], g, f"{region} | record coordinates", len(xo), "matched")
            xc = R.add_land(v[mask & v.visit_id.isin(xo.visit_id)].reset_index(drop=True), land, r)
            tc = terms_for(xc, False)
            rob, _ = S.fit(xc, g, tc, pd.factorize(xc.site_id)[0])
            rows += S.rows_from(rob, ["log_fsize", "edge_crop", "crop"], g, f"{region} | cell centre", len(xc), "matched")
    d = pd.DataFrame(rows); d.to_csv(out / "obs_vs_cell_matched.csv", index=False); return d


def accuracy_subsets(out):
    """Cell-centre models on: all visits; visits with >=1 record of known accuracy <=100 m;
    visits whose records ALL have known accuracy <=100 m. Isolates visit selection."""
    v, land = R.load("results_ext/data_pooled", [2024, 2025])
    rec = pd.concat([pd.read_csv(D / "records_nypa.csv")[["visit_id", "pos_acc_m"]],
                     pd.read_csv(D / "records_cb.csv")[["visit_id", "pos_acc_m"]]])
    good = rec.pos_acc_m.notna() & (rec.pos_acc_m <= 100)
    any_good = set(rec.loc[good, "visit_id"])
    all_good = set(rec.visit_id) - set(rec.loc[~good, "visit_id"])
    rows = []
    for g, r in S.EXT_R.items():
        for region, mask in [("NY-PA", v.state.isin(["NY", "PA"])), ("Corn Belt", ~v.state.isin(["NY", "PA"])), ("Six states", v.state.notna())]:
            for sub, keep in [("all visits", None), (">=1 record <=100 m", any_good), ("all records <=100 m", all_good)]:
                vv = v[mask] if keep is None else v[mask & v.visit_id.isin(keep)]
                x = R.add_land(vv.reset_index(drop=True), land, r); t = terms_for(x, False)
                rob, _ = S.fit(x, g, t, pd.factorize(x.site_id)[0])
                rows += S.rows_from(rob, ["log_fsize", "edge_crop", "crop"], g, f"{region} | {sub}", len(x), "accuracy subsets")
    d = pd.DataFrame(rows); d.to_csv(out / "accuracy_subsets.csv", index=False); return d


def accurate_with_composition(out):
    """Accurately located visits (all records <=100 m), cell centre, + WorldCover composition."""
    import composition_models as C
    v, land = R.load("results_ext/data_pooled", [2024, 2025])
    wc = pd.read_csv("results_ext/data_pooled/worldcover_composition.csv")
    rec = pd.concat([pd.read_csv(D / "records_nypa.csv")[["visit_id", "pos_acc_m"]],
                     pd.read_csv(D / "records_cb.csv")[["visit_id", "pos_acc_m"]]])
    good = rec.pos_acc_m.notna() & (rec.pos_acc_m <= 100)
    all_good = set(rec.visit_id) - set(rec.loc[~good, "visit_id"])
    rows = []
    for g, r in S.EXT_R.items():
        for region, mask in [("NY-PA", v.state.isin(["NY", "PA"])), ("Corn Belt", ~v.state.isin(["NY", "PA"])), ("Six states", v.state.notna())]:
            x = R.add_land(v[mask & v.visit_id.isin(all_good)].reset_index(drop=True), land, r)
            x = C.add_wc(x, wc, r); t = terms_for(x, False)
            for lab, extra in C.SPECS.items():
                rob, _ = S.fit(x, g, t + extra, pd.factorize(x.site_id)[0])
                rows += S.rows_from(rob, ["log_fsize", "edge_crop", "crop"] + extra, g, f"{region} | all records <=100 m {lab}", len(x), "accurate + composition")
    d = pd.DataFrame(rows); d.to_csv(out / "accurate_with_composition.csv", index=False); return d
