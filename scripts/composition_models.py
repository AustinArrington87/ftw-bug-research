"""Reviewer request: land-cover composition (ESA WorldCover 2021) as covariates."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
import reporting_rate_models as R
import reviewer_sensitivity as S

SPECS = {"+ semi-natural & developed": ["wc_seminatural", "wc_built"],
         "+ forest, grass/shrub, wetland, developed": ["wc_tree", "wc_grassshrub", "wc_wetland", "wc_built"]}


def add_wc(x, wc, radius):
    w = wc[wc.radius_m == radius].copy(); w["wc_grassshrub"] = w.wc_grass + w.wc_shrub
    cols = ["wc_seminatural", "wc_built", "wc_tree", "wc_grassshrub", "wc_wetland"]
    x = x.merge(w[["site_id"] + cols], on="site_id", how="left")
    for c in cols: x[c] = R.zs(x[c])
    return x


def vif(x, cols):
    X = sm.add_constant(x[cols].astype(float)).to_numpy(); out = {}
    for i, c in enumerate(cols, start=1):
        y = X[:, i]; o = np.delete(X, i, axis=1)
        r2 = 1 - ((y - o @ np.linalg.lstsq(o, y, rcond=None)[0]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = 1 / (1 - r2)
    return out


def run(data_dir, radii, part, extension):
    v, land = R.load(data_dir, [2024, 2025]); wc = pd.read_csv(Path(data_dir) / "worldcover_composition.csv")
    rows, vifs = [], []
    show = S.INTER if extension else ["log_fsize", "edge_crop", "crop"]
    for g, r in radii.items():
        x, terms = S.prep(v.copy(), land, g, r, extension); x = add_wc(x, wc, r)
        for label, extra in [("baseline", [])] + list(SPECS.items()):
            t = terms + extra
            rob, _ = S.fit(x, g, t, pd.factorize(x.site_id)[0])
            rows += S.rows_from(rob, show + extra if label != "baseline" else show, g, label, len(x), part)
            if extra:
                vifs.append({"part": part, "guild": g, "spec": label, **{k: round(val, 2) for k, val in vif(x, ["crop", "log_fsize", "edge_crop"] + extra).items()}})
    return rows, vifs


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    a, va = run("data_v3", S.BEST, "primary NY-PA", False)
    b, vb = run("results_ext/data_pooled", S.EXT_R, "extension six states", True)
    pd.DataFrame(a + b).to_csv(out / "composition_models.csv", index=False)
    pd.DataFrame(va + vb).to_csv(out / "composition_vif.csv", index=False)
    print(pd.DataFrame(va + vb).to_string())
