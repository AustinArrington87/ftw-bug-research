"""
Robustness checks for the primary models (2024-25, best radius per guild):
  baseline | no fields < 0.1 ha | top-10 observers removed | NY only | PA only

  python robustness_checks.py DATA_DIR OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import reporting_rate_models as R

BEST = {"bumble_bee": 2000, "other_bee": 3000, "syrphid": 1000}
TOP_OBS = 10


def coefs(x, guild, terms, label):
    x = x.dropna(subset=R.LAND).reset_index(drop=True)  # sites with no fields left
    _, rob = R.fit(x, guild, terms)
    ci = np.exp(rob.conf_int())
    return [{"guild": guild, "check": label, "term": t, "visits": len(x),
             "odds_ratio": np.exp(rob.params[t]), "or_lo95": ci.loc[t, 0],
             "or_hi95": ci.loc[t, 1], "p": rob.pvalues[t]} for t in R.LAND]


def main(data_dir, out_dir):
    v, land = R.load(data_dir, [2024, 2025])
    land_ns = pd.read_csv(Path(data_dir) / "landscape_avg_nosliver.csv")
    top = v.observer.value_counts().head(TOP_OBS).index
    print(f"top {TOP_OBS} observers hold {v.observer.isin(top).mean():.1%} of visits")
    rows = []
    for g, r in BEST.items():
        terms = R.LAND + R.CTRL
        base = R.add_land(v, land, r)
        rows += coefs(base, g, terms, "baseline")
        rows += coefs(R.add_land(v, land_ns, r), g, terms, "no fields <0.1 ha")
        rows += coefs(R.add_land(v[~v.observer.isin(top)].reset_index(drop=True), land, r),
                      g, terms, f"top {TOP_OBS} observers removed")
        no_pa = [t for t in terms if t != "pa"]
        for st in ["NY", "PA"]:
            sub = v[v.state == st].reset_index(drop=True)
            rows += coefs(R.add_land(sub, land, r), g, no_pa, f"{st} only")
    out = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(out_dir) / "robustness_2024_2025.csv", index=False)
    return out


if __name__ == "__main__":
    o = main(sys.argv[1], sys.argv[2])
    print(o.pivot_table(index=["guild", "check"], columns="term", values="odds_ratio")
          .round(3).to_string())
    print(o[o.p < 0.05][["guild", "check", "term", "odds_ratio", "or_lo95", "or_hi95", "p"]]
          .round(3).to_string())
