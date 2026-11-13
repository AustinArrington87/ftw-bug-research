"""
Peer-review revision, Corn Belt part (run in the folder that holds cornbelt/).

1. records_cb.csv: every record that entered the original Corn Belt build,
   with its site_id, visit_id, positional accuracy, and projected x/y.
   (Lets the analysis re-run with stricter accuracy thresholds.)
2. obs_metrics_cb_<year>.csv: FTW landscape metrics computed at each record's
   own coordinates (records with known accuracy <= 100 m), per state-year,
   at 1, 2, 3 km. Same metric code as the published analysis.

    python cornbelt_precise.py
"""
import numpy as np
import pandas as pd
import geopandas as gpd

import ftw_metrics_lean as L
import inat_pollinator_sites as m

W = "cornbelt"
STATES = ["IA", "IL", "IN", "OH"]


def records():
    df = pd.read_csv(f"{W}/obs_cornbelt.csv").drop(columns=["state"], errors="ignore")
    infra = df["rank"].isin(["subspecies", "variety", "form"])
    df.loc[infra, "species"] = df.loc[infra, "species"].str.split().str[:2].str.join(" ")
    df.loc[infra, "rank"] = "species"
    df = df[(df["rank"] == "species") & df.date.notna()]
    df = df[df.pos_acc_m.isna() | (df.pos_acc_m <= m.MAX_POS_ACC_M)]
    df["genus"] = df.species.str.split().str[0]
    df["guild"] = df.genus.map(m.guild)
    df = df[df.guild != "other"]
    df["year"] = df.date.str[:4].astype(int)
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    b = gpd.read_file(f"{W}/cornbelt_states.gpkg").to_crs("EPSG:4326")
    g = g[g.within(b.union_all())]
    g = gpd.sjoin(g, b[["STUSPS", "geometry"]], how="left", predicate="within").drop(columns="index_right")
    g = g.rename(columns={"STUSPS": "state"}).to_crs("EPSG:5070")
    cx = (g.geometry.x // 1000).astype(int); cy = (g.geometry.y // 1000).astype(int)
    g["site_id"] = "c" + cx.astype(str) + "_" + cy.astype(str)
    g["visit_id"] = g.site_id + "|" + g.observer.astype(str) + "|" + g.date
    g["x"] = g.geometry.x; g["y"] = g.geometry.y
    cols = ["obs_id", "date", "year", "observer", "species", "guild", "state", "pos_acc_m",
            "site_id", "visit_id", "x", "y"]
    return pd.DataFrame(g[cols])


def main():
    r = records()
    r.to_csv(f"{W}/upload_rev/records_cb.csv", index=False)
    print("records:", len(r), "| check vs build visits:",
          r.visit_id.nunique(), "unique visits (should equal visits_cb rows)", flush=True)
    p = r[r.pos_acc_m.notna() & (r.pos_acc_m <= 100)].copy()
    p["pid"] = p.x.round(0).astype(int).astype(str) + "_" + p.y.round(0).astype(int).astype(str)
    for yr in (2024, 2025):
        parts = []
        for st in STATES:
            pts = p[p.state == st].drop_duplicates("pid")
            if pts.empty:
                continue
            F = L.load(f"{W}/ftw/ftw_{st}_{yr}.parquet")
            out = L.metrics(F, pts[["x", "y"]].to_numpy(), pts.pid, [1000, 2000, 3000])
            parts.append(out); print(f"{st} {yr}: {len(pts)} points", flush=True)
            del F
        pd.concat(parts).to_csv(f"{W}/upload_rev/obs_metrics_cb_{yr}.csv", index=False)


if __name__ == "__main__":
    import os
    os.makedirs(f"{W}/upload_rev", exist_ok=True)
    main()
