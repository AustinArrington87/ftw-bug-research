"""
Declared secondary analysis (fixed before fitting; see Methods):
Bombus species with >=30 detections in 2024-25 agricultural visits, plus
all Bombus except B. impatiens. Primary reporting-rate model at 2 km.
Holm adjustment across the 8 responses for field size and edge terms.

  python secondary_bombus_species.py DATA_DIR OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import reporting_rate_models as R

SPECIES = ["Bombus impatiens", "Bombus griseocollis", "Bombus bimaculatus",
           "Bombus perplexus", "Bombus ternarius", "Bombus fervidus",
           "Bombus vagans"]
RADIUS = 2000


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def run(data_dir, years):
    v, land = R.load(data_dir, years)
    det = pd.read_csv(Path(data_dir) / "detections.csv")
    det = det[det.visit_id.isin(v.visit_id)]
    resp = {}
    for sp in SPECIES:
        resp[sp] = v.visit_id.isin(det.loc[det.species == sp, "visit_id"]).astype(int)
    non_imp = det[(det.guild == "bumble_bee") & (det.species != "Bombus impatiens")]
    resp["Bombus excl. impatiens"] = v.visit_id.isin(non_imp.visit_id).astype(int)
    x = R.add_land(v, land, RADIUS)
    rows = []
    for name, y in resp.items():
        x["_y"] = y.values
        _, rob = R.fit(x, "_y", R.LAND + R.CTRL)
        ci = np.exp(rob.conf_int())
        for t in R.LAND:
            rows.append({"response": name, "detections": int(y.sum()), "term": t,
                         "odds_ratio": np.exp(rob.params[t]), "or_lo95": ci.loc[t, 0],
                         "or_hi95": ci.loc[t, 1], "p": rob.pvalues[t]})
    out = pd.DataFrame(rows)
    for t in ["log_fsize", "edge_crop"]:  # declared Holm family
        m = out.term == t
        out.loc[m, "p_holm"] = holm(out.loc[m, "p"])
    return out


if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
    for yrs, tag in [([2024, 2025], "2024_2025"), (list(range(2017, 2024)), "2017_2023")]:
        r = run(data_dir, yrs)
        r.to_csv(out_dir / f"secondary_bombus_species_{tag}.csv", index=False)
        print(tag); print(r.round(3).to_string())
