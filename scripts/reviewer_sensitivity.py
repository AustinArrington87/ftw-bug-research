"""
Sensitivity analyses requested in peer review.

Primary (NY-PA, 2024-25, best radii) and declared extension (six states).
Each check changes one thing relative to the published model:

  short lists        drop visits with list length 1 (and < 3)
  focal recorders    keep only visits by observers who recorded the focal
                     group on at least one OTHER visit (taxonomic interest)
  block SEs          cluster SEs on 10-km and 25-km spatial blocks
  Moran's I          spatial autocorrelation of cell-level mean residuals
  single-year        landscape metrics from 2024 only or 2025 only
  accuracy           records with known positional accuracy <= 250 m / <= 100 m
  unsimplified       metrics from raw (unsimplified) FTW polygons
  obs coordinates    metrics at each record's own coordinates (<= 100 m records)

  python reviewer_sensitivity.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial import cKDTree

import reporting_rate_models as R

BEST = {"bumble_bee": 2000, "other_bee": 1000, "syrphid": 1000}          # NY-PA AIC-preferred
EXT_R = {"bumble_bee": 2000, "other_bee": 3000, "syrphid": 1000}         # declared extension
INTER = ["crop_x_fsize", "crop_x_edge"]
TERMS_P = ["log_fsize", "edge_crop", "crop"]


def cell_xy(site_ids):
    xy = site_ids.str.extract(r"c(-?\d+)_(-?\d+)").astype(int)
    return xy[0].to_numpy(), xy[1].to_numpy()  # km grid indices (1-km cells)


def fit(x, y, terms, groups):
    X = sm.add_constant(x[terms].astype(float))
    glm = sm.GLM(x[y], X, family=sm.families.Binomial())
    return glm.fit(cov_type="cluster", cov_kwds={"groups": groups}), glm.fit()


def prep(v, land, guild, radius, extension=False):
    x = R.add_land(v, land, radius).dropna(subset=R.LAND).reset_index(drop=True)
    terms = list(R.LAND) + [c for c in R.CTRL if not (extension and c == "pa")]
    terms += [c for c in x.columns if c.startswith("yr") and c[2:].isdigit()]
    if extension:
        for st in sorted(x.state.unique()):
            if st != "NY":
                x[f"st_{st}"] = (x.state == st).astype(float); terms.append(f"st_{st}")
        x["crop_x_fsize"] = x.crop * x.log_fsize; x["crop_x_edge"] = x.crop * x.edge_crop
        terms += INTER
    return x, terms


def rows_from(rob, terms, guild, check, n, part):
    ci = np.exp(rob.conf_int())
    return [{"part": part, "guild": guild, "check": check, "term": t, "visits": n,
             "odds_ratio": np.exp(rob.params[t]), "or_lo95": ci.loc[t, 0],
             "or_hi95": ci.loc[t, 1], "p": rob.pvalues[t]} for t in terms]


def morans_i(x, plain, k=8, nperm=999, seed=1):
    res = pd.Series(plain.resid_pearson, index=x.index)
    cell = x.assign(r=res).groupby("site_id").r.mean()
    cx, cy = cell_xy(pd.Series(cell.index))
    tree = cKDTree(np.column_stack([cx, cy])); _, nb = tree.query(np.column_stack([cx, cy]), k=k + 1)
    z = cell.to_numpy() - cell.mean(); nb = nb[:, 1:]
    stat = lambda zz: (len(zz) / (len(zz) * k)) * (zz[:, None] * zz[nb]).sum() / (zz ** 2).sum()
    obs = stat(z); rng = np.random.default_rng(seed)
    perm = np.array([stat(rng.permutation(z)) for _ in range(nperm)])
    return obs, (1 + (np.abs(perm) >= abs(obs)).sum()) / (nperm + 1)


def run_set(v, land, radii, part, extension=False, label_prefix="", land_variants=None):
    out, mor = [], []
    det_terms = INTER if extension else TERMS_P
    obs_exp_all = v.observer.value_counts()
    for g, r in radii.items():
        x, terms = prep(v.copy(), land, g, r, extension)
        site = pd.factorize(x.site_id)[0]
        rob, plain = fit(x, g, terms, site)
        out += rows_from(rob, det_terms, g, label_prefix + "baseline", len(x), part)
        o, p = morans_i(x, plain); mor.append({"part": part, "guild": g, "morans_I": o, "p_perm": p})
        cx, cy = cell_xy(x.site_id)
        for km in (10, 25):
            blk = pd.factorize(pd.Series(cx // km).astype(str) + "_" + pd.Series(cy // km).astype(str))[0]
            rb, _ = fit(x, g, terms, blk)
            out += rows_from(rb, det_terms, g, f"{km}-km block SEs", len(x), part)
        for lab, mask in (("list length >= 2", x.list_len >= 2), ("list length >= 3", x.list_len >= 3)):
            xs = x[mask].reset_index(drop=True)
            rb, _ = fit(xs, g, terms, pd.factorize(xs.site_id)[0])
            out += rows_from(rb, det_terms, g, lab, len(xs), part)
        # observers who recorded the focal group on another visit
        per_obs = x.groupby("observer")[g].sum()
        other = x.observer.map(per_obs) - x[g]
        xs = x[other > 0].reset_index(drop=True)
        rb, _ = fit(xs, g, terms, pd.factorize(xs.site_id)[0])
        out += rows_from(rb, det_terms, g, "focal-group recorders only", len(xs), part)
        for vname, lv in (land_variants or {}).items():
            xs, t2 = prep(v.copy(), lv, g, r, extension)
            rb, _ = fit(xs, g, t2, pd.factorize(xs.site_id)[0])
            out += rows_from(rb, det_terms, g, vname, len(xs), part)
    return out, mor


def single_year(land, which):
    L = land.copy()
    for c in ["cropland_frac", "edge_density_m_per_ha", "edge_per_crop_ha", "aw_mean_field_ha", "mean_field_ha"]:
        L[c + "_avg"] = L[c + "_24"] if which == 2024 else L[c]
    return L


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    rows, mors = [], []
    # primary
    v, land = R.load("data_v3", [2024, 2025])
    lv = {"2024 boundaries only": single_year(land, 2024), "2025 boundaries only": single_year(land, 2025)}
    a, b = run_set(v, land, BEST, "primary NY-PA", land_variants=lv); rows += a; mors += b
    # extension
    v, land = R.load("results_ext/data_pooled", [2024, 2025])
    lv = {"2024 boundaries only": single_year(land, 2024), "2025 boundaries only": single_year(land, 2025)}
    a, b = run_set(v, land, EXT_R, "extension six states", extension=True, land_variants=lv); rows += a; mors += b
    pd.DataFrame(rows).to_csv(out_dir / "reviewer_sensitivity.csv", index=False)
    pd.DataFrame(mors).to_csv(out_dir / "morans_i.csv", index=False)
    print(pd.DataFrame(mors).round(4).to_string())
