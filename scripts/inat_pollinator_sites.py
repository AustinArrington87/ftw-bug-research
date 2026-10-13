"""
Pull ESNPS (and optional target-group bee) observations from iNaturalist and
build sites + visits for occupancy modelling.

  python inat_pollinator_sites.py pull  --out obs_raw.csv
  python inat_pollinator_sites.py build --obs obs_raw.csv --cell-m 1000 \
         --sites sites.csv --visits visits.csv --detections detections.csv

Outputs
  sites.csv       site_id, lat, lon, n_visits, n_observers, n_obs, n_species
                  (feeds ftw_landscape_metrics.py metrics --sites)
  visits.csv      site_id, visit_id (observer x date), date, observer, n_obs
  detections.csv  site_id, visit_id, species   (one row per detection)

A "site" is a grid cell; a "visit" is one observer on one date in that cell.
Visits give the sampling effort an occupancy model needs.
"""
import argparse
import time

import pandas as pd
import requests

API = "https://api.inaturalist.org/v1/observations"
ESNPS_PROJECT = "empire-state-native-pollinator-survey"
MAX_POS_ACC_M = 1000  # drop records with worse location precision
NY_BBOX = (-79.8, 40.4, -71.8, 45.1)  # coarse; use --boundary for a true state clip

# First-pass guild assignment by genus. Replace with iNat taxon ancestry
# (family-level) before final analysis.
BEE_GENERA = {
    "Bombus", "Apis", "Xylocopa", "Halictus", "Augochlora", "Augochlorella",
    "Augochloropsis", "Agapostemon", "Lasioglossum", "Andrena", "Osmia",
    "Megachile", "Ceratina", "Colletes", "Hylaeus", "Melissodes", "Anthophora",
    "Nomada", "Eucera", "Peponapis", "Habropoda", "Sphecodes", "Calliopsis",
    "Dufourea", "Hoplitis", "Heriades", "Anthidium", "Coelioxys", "Triepeolus",
    "Epeolus", "Svastra", "Perdita", "Panurginus", "Pseudopanurgus", "Macropis",
    "Holcopasites", "Stelis", "Chelostoma", "Dianthidium", "Anthidiellum",
    "Melitta", "Cemolobus",
}
SYRPHID_GENERA = {
    "Toxomerus", "Eristalis", "Syritta", "Syrphus", "Helophilus",
    "Sphaerophoria", "Allograpta", "Temnostoma", "Milesia", "Spilomyia",
    "Eupeodes", "Chrysotoxum", "Platycheirus", "Melanostoma", "Volucella",
    "Microdon", "Mallota", "Palpada", "Sericomyia", "Xylota", "Chalcosyrphus",
    "Criorhina", "Pipiza", "Paragus", "Ocyptamus", "Copestylum", "Merodon",
    "Eristalinus", "Parhelophilus", "Lejops", "Orthonevra", "Neoascia",
    "Cheilosia", "Dasysyrphus", "Epistrophe",
}


def guild(genus):
    if genus == "Bombus":
        return "bumble_bee"
    if genus in BEE_GENERA:
        return "other_bee"
    if genus in SYRPHID_GENERA:
        return "syrphid"
    return "other"


def pull(out, project=ESNPS_PROJECT, taxon_id=None, place_id=None,
         d1="2017-01-01", d2="2021-12-31", bbox=None):
    params = {"quality_grade": "research", "per_page": 200, "order_by": "id",
              "order": "asc", "d1": d1, "d2": d2, "id_above": 0}
    if project:
        params["project_id"] = project
    if taxon_id:
        params["taxon_id"] = taxon_id   # e.g. Anthophila (bees); verify id on iNat
    if place_id:
        params["place_id"] = place_id   # e.g. Pennsylvania; verify id on iNat
    if bbox:  # W S E N; clip to exact state polygons later with build --boundary
        params.update(swlng=bbox[0], swlat=bbox[1], nelng=bbox[2], nelat=bbox[3])
    rows = []
    while True:
        r = requests.get(API, params=params, timeout=60)
        r.raise_for_status()
        res = r.json()["results"]
        if not res:
            break
        for o in res:
            gj, t = o.get("geojson") or {}, o.get("taxon") or {}
            if not gj or o.get("obscured") or o.get("taxon_geoprivacy") == "obscured":
                continue  # obscured coords can be offset by tens of km
            rows.append({
                "obs_id": o["id"], "date": o.get("observed_on"),
                "observer": (o.get("user") or {}).get("login"),
                "lon": gj["coordinates"][0], "lat": gj["coordinates"][1],
                "pos_acc_m": o.get("positional_accuracy"),
                "species": t.get("name"), "rank": t.get("rank"),
            })
        params["id_above"] = res[-1]["id"]
        print(f"{len(rows)} kept, last id {params['id_above']}")
        time.sleep(1)  # be polite: iNat asks for <=1 req/sec
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


def build(obs_path, cell_m, sites_out, visits_out, det_out,
          boundary=None, bbox=NY_BBOX, keep_other=False):
    import geopandas as gpd

    df = pd.read_csv(obs_path)
    # Roll infraspecific IDs up to species (e.g. Xylocopa virginica virginica)
    infra = df["rank"].isin(["subspecies", "variety", "form"])
    df.loc[infra, "species"] = df.loc[infra, "species"].str.split().str[:2].str.join(" ")
    df.loc[infra, "rank"] = "species"
    df = df[(df["rank"] == "species") & df["date"].notna()]
    df = df[df["pos_acc_m"].isna() | (df["pos_acc_m"] <= MAX_POS_ACC_M)]
    if not boundary:  # coarse fallback only; a boundary file replaces it
        w, s_, e, n = bbox
        df = df[df.lon.between(w, e) & df.lat.between(s_, n)]
    df["genus"] = df["species"].str.split().str[0]
    df["guild"] = df["genus"].map(guild)
    if not keep_other:
        df = df[df["guild"] != "other"]
    df["year"] = df["date"].str[:4].astype(int)
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),
                         crs="EPSG:4326")
    if boundary:  # e.g. Census TIGER state polygons (any CRS; reprojected)
        b = gpd.read_file(boundary).to_crs("EPSG:4326")
        g = g[g.within(b.union_all())]
        if "STUSPS" in b.columns:  # tag each record with its state
            g = gpd.sjoin(g, b[["STUSPS", "geometry"]], how="left",
                          predicate="within").drop(columns="index_right")
            g = g.rename(columns={"STUSPS": "state"})
    g = g.to_crs("EPSG:5070")
    cx = (g.geometry.x // cell_m).astype(int)
    cy = (g.geometry.y // cell_m).astype(int)
    g["site_id"] = "c" + cx.astype(str) + "_" + cy.astype(str)
    g["visit_id"] = g["site_id"] + "|" + g["observer"].astype(str) + "|" + g["date"]

    det = g[["site_id", "visit_id", "year", "species", "genus", "guild"]].drop_duplicates()
    visits = (g.groupby(["site_id", "visit_id", "year", "date", "observer"])
               .size().rename("n_obs").reset_index())

    centers = gpd.GeoDataFrame(
        {"site_id": g["site_id"]},
        geometry=gpd.points_from_xy((cx + 0.5) * cell_m, (cy + 0.5) * cell_m),
        crs="EPSG:5070").drop_duplicates("site_id").to_crs("EPSG:4326")
    sites = centers.assign(lat=centers.geometry.y, lon=centers.geometry.x)
    agg_extra = {"state": ("state", lambda x: x.mode().iat[0])} if "state" in g else {}
    summ = g.groupby("site_id").agg(**agg_extra,
        n_visits=("visit_id", "nunique"), n_observers=("observer", "nunique"),
        n_obs=("species", "size"), n_species=("species", "nunique"))
    sites = sites.drop(columns="geometry").merge(summ, on="site_id")

    sites.to_csv(sites_out, index=False)
    visits.to_csv(visits_out, index=False)
    det.to_csv(det_out, index=False)
    print(f"{len(sites)} sites, {len(visits)} visits, {len(det)} detections")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pull")
    a.add_argument("--out", required=True)
    a.add_argument("--project", default=ESNPS_PROJECT)
    a.add_argument("--taxon-id", type=int)
    a.add_argument("--place-id", type=int)
    a.add_argument("--d1", default="2017-01-01")
    a.add_argument("--d2", default="2021-12-31")
    a.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"))
    b = sub.add_parser("build")
    b.add_argument("--obs", required=True)
    b.add_argument("--cell-m", type=int, default=1000)
    b.add_argument("--sites", default="sites.csv")
    b.add_argument("--visits", default="visits.csv")
    b.add_argument("--detections", default="detections.csv")
    b.add_argument("--boundary", help="polygon file to clip to (e.g. NY state)")
    b.add_argument("--keep-other", action="store_true",
                   help="keep taxa outside bee/syrphid guilds")
    args = p.parse_args()
    if args.cmd == "pull":
        pull(args.out, args.project or None, args.taxon_id, args.place_id,
             args.d1, args.d2, args.bbox)
    else:
        build(args.obs, args.cell_m, args.sites, args.visits, args.detections,
              boundary=args.boundary, keep_other=args.keep_other)


if __name__ == "__main__":
    main()
