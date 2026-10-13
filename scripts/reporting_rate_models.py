"""
Reporting-rate models: P(focal guild recorded on a visit) vs FTW landscape
structure, with list length as the effort covariate (Szabo et al. 2010).

Replaces guild-level occupancy, which was not identifiable here: 69% of
site-years had a single visit, so occupancy and detection could not be
separated (psi estimates ran to ~0.97 with very wide intervals).

  logit P(y_ij=1) = landscape_i + state + year + doy + doy^2
                    + log(list length_j) + log(observer experience_j)

Binomial GLM; site-clustered (cluster-robust) SEs for inference, plain GLM
log-likelihood for AIC. Every radius uses the identical set of visits, so
AIC is comparable across radii.

  python reporting_rate_models.py DATA_DIR OUT_DIR [--years 2024 2025]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

SEASON = (91, 304)
CROP_MIN = 0.05
RADII = [1000, 2000, 3000]
GUILDS = ["bumble_bee", "other_bee", "syrphid"]
LAND = ["crop", "log_fsize", "edge_crop"]
CTRL = ["pa", "doy", "doy2", "log_list", "obs_exp"]  # + one dummy per year after the first


def zs(x):
    return (x - x.mean()) / x.std()


def load(data_dir, years):
    d = Path(data_dir)
    land = pd.read_csv(d / "landscape_avg.csv")
    sites = pd.read_csv(d / "sites.csv")[["site_id", "state"]]
    visits = pd.read_csv(d / "visits.csv")
    det = pd.read_csv(d / "detections.csv")

    # common site set: >=5% cropland at EVERY radius (and metrics defined), so all
    # radii use identical visits and edge per cropland ha never has a near-zero
    # denominator (at <5% cropland the ratio is dominated by clipped slivers)
    ok_r = (land[land.radius_m.isin(RADII) & (land.cropland_frac_avg >= CROP_MIN)]
            .dropna(subset=["aw_mean_field_ha_avg", "edge_per_crop_ha_avg"])
            .groupby("site_id").radius_m.nunique())
    ok = set(ok_r[ok_r == len(RADII)].index)

    obs_exp = visits.groupby("observer").n_obs.sum()
    v = visits[visits.year.isin(years) & visits.site_id.isin(ok)].copy()
    v["doy_raw"] = pd.to_datetime(v.date).dt.dayofyear
    v = v[v.doy_raw.between(*SEASON)].merge(sites, on="site_id")
    ll = det.groupby("visit_id").species.nunique()
    v["list_len"] = v.visit_id.map(ll).fillna(0)
    v = v[v.list_len > 0]
    v["log_list"] = zs(np.log(v.list_len))
    v["obs_exp"] = zs(np.log(v.observer.map(obs_exp)))
    v["doy"] = zs(v.doy_raw.astype(float))
    v["doy2"] = v.doy ** 2
    v["pa"] = (v.state == "PA").astype(float)
    for yr in sorted(years)[1:]:
        v[f"yr{yr}"] = (v.year == yr).astype(float)
    for g in GUILDS:
        v[g] = v.visit_id.isin(det.loc[det.guild == g, "visit_id"]).astype(int)
    return v.reset_index(drop=True), land


def add_land(v, land, radius):
    L = land[land.radius_m == radius][["site_id", "cropland_frac_avg",
                                       "aw_mean_field_ha_avg", "edge_per_crop_ha_avg"]]
    x = v.merge(L, on="site_id", how="left")
    x["crop"] = zs(x.cropland_frac_avg)
    x["log_fsize"] = zs(np.log(x.aw_mean_field_ha_avg))
    x["edge_crop"] = zs(x.edge_per_crop_ha_avg)
    return x


def year_terms(x):
    return [c for c in x.columns if c.startswith("yr") and c[2:].isdigit()]


def fit(x, guild, terms):
    terms = list(terms) + [t for t in year_terms(x) if t not in terms]
    X = sm.add_constant(x[terms].astype(float))
    glm = sm.GLM(x[guild], X, family=sm.families.Binomial())
    plain = glm.fit()
    groups = pd.factorize(x.site_id)[0]
    robust = glm.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return plain, robust


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir"); p.add_argument("out_dir")
    p.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    v, land = load(a.data_dir, a.years)
    print(f"{len(v)} visits, {v.site_id.nunique()} sites, years {a.years}", flush=True)

    sel, coefs = [], []
    for g in GUILDS:
        for r in RADII:
            x = add_land(v, land, r)
            plain, rob = fit(x, g, LAND + CTRL)
            sel.append({"guild": g, "radius_m": r, "visits": len(x),
                        "sites": x.site_id.nunique(), "prevalence": x[g].mean(),
                        "AIC": plain.aic})
            ci = rob.conf_int()
            coefs.append(pd.DataFrame({
                "guild": g, "radius_m": r, "term": rob.params.index,
                "est": rob.params.values, "se_cluster": rob.bse.values,
                "lo95": ci[0].values, "hi95": ci[1].values, "p": rob.pvalues.values,
                "odds_ratio": np.exp(rob.params.values),
                "or_lo95": np.exp(ci[0].values), "or_hi95": np.exp(ci[1].values)}))
    s = pd.DataFrame(sel)
    s["dAIC"] = s.AIC - s.groupby("guild").AIC.transform("min")
    tag = "_".join(map(str, a.years))
    s.to_csv(out / f"rr_model_selection_{tag}.csv", index=False)
    pd.concat(coefs).to_csv(out / f"rr_coefficients_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
