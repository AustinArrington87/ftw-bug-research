"""
Single-season occupancy models (MacKenzie et al. 2002) for pollinator guilds
against FTW landscape structure, fit by maximum likelihood.

Units      : site x year (closure assumed within a season, Apr-Oct)
Visits     : one observer on one date in a 1-km cell (target-group visits:
             any bee or hoverfly recorded), so a visit without the focal guild
             is a non-detection (van Strien et al. 2013 approach)
Occupancy  : cropland fraction, log area-weighted field size, edge per ha of
             cropland, state (PA vs NY), year
Detection  : day of year (+ quadratic), log observer experience

  python occupancy_models.py DATA_DIR OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log1p
from statsmodels.tools.numdiff import approx_hess

SEASON = (91, 304)  # Apr 1 - Oct 31 (day of year)
CROP_MIN = 0.05     # agricultural landscapes: >=5% cropland at 2 km
OCC_VARS = ["crop", "log_fsize", "edge_crop", "pa", "y2025"]
DET_VARS = ["doy", "doy2", "obs_exp"]


def zs(x):
    return (x - x.mean()) / x.std()


def prepare(data_dir, radius, years=(2024, 2025), guild="bumble_bee", sites_at=2000):
    d = Path(data_dir)
    land = pd.read_csv(d / "landscape_avg.csv")
    sites = pd.read_csv(d / "sites.csv")[["site_id", "state"]]
    visits = pd.read_csv(d / "visits.csv")
    det = pd.read_csv(d / "detections.csv")

    # fixed site set (>=5% cropland at 2 km) so radii are compared on the same units
    keep = land[(land.radius_m == sites_at) & (land.cropland_frac_avg >= CROP_MIN)].site_id
    L = land[(land.radius_m == radius) & land.site_id.isin(keep)].copy()
    L = L.dropna(subset=["aw_mean_field_ha_avg", "edge_per_crop_ha_avg"])

    obs_exp = visits.groupby("observer").n_obs.sum()  # experience across full dataset
    v = visits[visits.year.isin(years) & visits.site_id.isin(L.site_id)].copy()
    v["doy"] = pd.to_datetime(v.date).dt.dayofyear
    v = v[v.doy.between(*SEASON)]
    v["obs_exp"] = np.log(v.observer.map(obs_exp))
    hit = set(det.loc[det.guild == guild, "visit_id"])
    v["y"] = v.visit_id.isin(hit).astype(int)
    v["unit"] = v.site_id + "|" + v.year.astype(str)

    u = (v.groupby("unit").agg(site_id=("site_id", "first"), year=("year", "first"))
         .reset_index().merge(L, on="site_id").merge(sites, on="site_id"))
    u["crop"] = zs(u.cropland_frac_avg)
    u["log_fsize"] = zs(np.log(u.aw_mean_field_ha_avg))
    u["edge_crop"] = zs(u.edge_per_crop_ha_avg)
    u["pa"] = (u.state == "PA").astype(float)
    u["y2025"] = (u.year == 2025).astype(float)
    u = u.sort_values("unit").reset_index(drop=True)

    v = v[v.unit.isin(u.unit)].sort_values("unit").reset_index(drop=True)
    v["doy"] = zs(v.doy.astype(float))
    v["doy2"] = v.doy ** 2
    v["obs_exp"] = zs(v.obs_exp)
    return u, v


class Occu:
    def __init__(self, u, v, occ_vars=OCC_VARS, det_vars=DET_VARS):
        self.Xo = np.column_stack([np.ones(len(u))] + [u[c].to_numpy(float) for c in occ_vars])
        self.Xd = np.column_stack([np.ones(len(v))] + [v[c].to_numpy(float) for c in det_vars])
        self.y = v.y.to_numpy()
        codes = pd.Categorical(v.unit, categories=u.unit, ordered=True).codes
        self.idx = codes
        self.det_any = np.bincount(codes, weights=self.y, minlength=len(u)) > 0
        self.names = (["psi:Intercept"] + [f"psi:{c}" for c in occ_vars]
                      + ["p:Intercept"] + [f"p:{c}" for c in det_vars])
        self.ko = self.Xo.shape[1]

    def nll(self, b):
        lpsi = self.Xo @ b[: self.ko]
        lp = self.Xd @ b[self.ko:]
        # log p and log(1-p), numerically stable
        log_p = -np.logaddexp(0, -lp)
        log_q = -np.logaddexp(0, lp)
        ll_det = np.bincount(self.idx, weights=np.where(self.y == 1, log_p, log_q),
                             minlength=len(lpsi))
        log_psi = -np.logaddexp(0, -lpsi)
        log_1mpsi = -np.logaddexp(0, lpsi)
        ll = np.where(self.det_any, log_psi + ll_det,
                      np.logaddexp(log_psi + ll_det, log_1mpsi))
        return -ll.sum()

    def fit(self):
        b0 = np.zeros(len(self.names))
        r = minimize(self.nll, b0, method="BFGS", options={"gtol": 1e-6, "maxiter": 5000})
        H = approx_hess(r.x, self.nll)
        se = np.sqrt(np.diag(np.linalg.inv(H)))
        self.res = r
        self.table = pd.DataFrame({"term": self.names, "est": r.x, "se": se})
        self.table["z"] = self.table.est / self.table.se
        self.table["lo95"] = self.table.est - 1.96 * se
        self.table["hi95"] = self.table.est + 1.96 * se
        self.aic = 2 * r.fun + 2 * len(r.x)
        return self


def main(data_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, tables = [], []
    for guild in ["bumble_bee", "other_bee", "syrphid"]:
        for r in [1000, 2000, 3000]:
            u, v = prepare(data_dir, r, guild=guild)
            m = Occu(u, v).fit()
            rows.append({"guild": guild, "radius_m": r, "units": len(u), "visits": len(v),
                         "naive_occ": m.det_any.mean(), "AIC": m.aic,
                         "converged": m.res.success})
            t = m.table.assign(guild=guild, radius_m=r); tables.append(t)
            print(guild, r, len(u), round(m.aic, 1), m.res.success, flush=True)
    s = pd.DataFrame(rows)
    s["dAIC"] = s.AIC - s.groupby("guild").AIC.transform("min")
    s.to_csv(out / "model_selection.csv", index=False)
    pd.concat(tables).to_csv(out / "coefficients.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
