"""
FTW landscape metrics around pollinator survey sites (NY–PA).

Step 1  pull   : download FTW Global field polygons for the study bbox
                 (anonymous S3 read, filtered by year + confidence) -> GeoParquet
Step 2  metrics: for each survey site and buffer radius, compute
                 field-size and field-edge-density metrics -> CSV

Usage
  python ftw_landscape_metrics.py pull    --year 2025 --out ftw_nypa_2025.parquet
  python ftw_landscape_metrics.py metrics --fields ftw_nypa_2025.parquet \
         --sites sites.csv --out site_metrics.csv

sites.csv needs columns: site_id, lat, lon  (WGS84)

Data: Fields of The World Global (CC-BY-4.0), Taylor Geospatial Institute /
Microsoft AI for Good. Cite Kerner et al. 2025 (AAAI) and the FTW Global release.
"""
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

# FTW Global vectors, partitioned by country (Hive layout). Verify path against
# https://data.source.coop/ftw/global-data/predictions/vectors/collection.json
FTW_US_GLOB = (
    "s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/predictions/"
    "vectors/alpha/results-by-admin-conf/admin:country_code=US/*.parquet"
)
NYPA_BBOX = (-80.6, 39.7, -71.8, 45.1)  # lon/lat: west, south, east, north
CONF_MIN = 69          # FTW's recommended reliability filter (0-100 scale)
EQUAL_AREA = "EPSG:5070"  # CONUS Albers, metres
RADII_M = [500, 1000, 2000, 3000]


def pull(year: int, out: str, bbox=NYPA_BBOX, conf_min=CONF_MIN):
    import duckdb

    con = duckdb.connect()
    for ext in ("httpfs", "spatial"):
        con.install_extension(ext)
        con.load_extension(ext)
    con.execute("SET s3_region='us-west-2'; SET s3_access_key_id=''; "
                "SET s3_secret_access_key='';")
    # Prediction timestamps are UTC. Without this, year() runs in local time
    # and a 2025-01-01T00:00Z stamp reads as 2024 in US timezones.
    con.execute("SET TimeZone='UTC';")
    # bucket name contains dots; virtual-host style breaks TLS
    con.execute("SET s3_url_style='path';")
    w, s, e, n = bbox
    # bbox struct predicate lets DuckDB skip row groups outside the study area.
    # Field names assume the GeoParquet 1.1 bbox covering (xmin/ymin/xmax/ymax);
    # run `DESCRIBE SELECT * FROM read_parquet(...) LIMIT 1` if this errors.
    q = f"""
    COPY (
      SELECT id, "metrics:area" AS area_m2, "metrics:perimeter" AS perim_m,
             confidence, "determination:datetime" AS det_dt, geometry
      FROM read_parquet('{FTW_US_GLOB}', hive_partitioning = true)
      WHERE bbox.xmax >= {w} AND bbox.xmin <= {e}
        AND bbox.ymax >= {s} AND bbox.ymin <= {n}
        AND year("determination:datetime") = {year}
        -- keep fields outside the confidence layer's coverage (NULL);
        -- they stay identifiable via the confidence column for sensitivity runs
        AND (confidence >= {conf_min} OR confidence IS NULL)
    ) TO '{out}' (FORMAT PARQUET);
    """
    con.execute(q)
    print(f"wrote {out}")


def load_fields(path: str) -> gpd.GeoDataFrame:
    df = pd.read_parquet(path)
    geom = df["geometry"]
    if not isinstance(geom.iloc[0], shapely.Geometry):
        geom = shapely.from_wkb(geom)
    gdf = gpd.GeoDataFrame(df.drop(columns="geometry"), geometry=geom, crs="EPSG:4326")
    gdf = gdf.to_crs(EQUAL_AREA)
    # Recompute in equal-area CRS so metrics are consistent and traceable
    gdf["area_m2"] = gdf.geometry.area
    gdf["perim_m"] = gdf.geometry.length
    return gdf


def site_metrics(fields: gpd.GeoDataFrame, sites: gpd.GeoDataFrame, radii=RADII_M):
    sindex = fields.sindex
    centroids = fields.geometry.representative_point()
    boundaries = fields.geometry.boundary
    rows = []
    for _, site in sites.iterrows():
        for r in radii:
            buf = site.geometry.buffer(r)
            buf_area = buf.area
            cand = sindex.query(buf, predicate="intersects")
            out = {"site_id": site["site_id"], "radius_m": r}
            if len(cand) == 0:
                rows.append(out | {"n_fields": 0, "cropland_frac": 0.0,
                                   "edge_density_m_per_ha": 0.0})
                continue
            sub = fields.iloc[cand]
            # Cropland share and edge length use geometry clipped to the buffer.
            # Edges = field boundaries inside the buffer only; the buffer's own
            # cut lines are never counted.
            clipped_area = sub.geometry.intersection(buf).area.sum()
            # union_all merges shared boundaries so each edge counts once
            # (FRAGSTATS convention); otherwise adjacent fields double-count.
            edge_len = shapely.union_all(
                boundaries.iloc[cand].intersection(buf).to_numpy()
            ).length
            # Field-size stats use whole fields whose interior point is in buffer
            inside = centroids.iloc[cand].within(buf).to_numpy()
            sizes_ha = sub.loc[inside, "area_m2"].to_numpy() / 1e4
            out |= {
                "n_fields": int(inside.sum()),
                "cropland_frac": clipped_area / buf_area,
                "edge_density_m_per_ha": edge_len / (buf_area / 1e4),
                "mean_field_ha": float(np.mean(sizes_ha)) if sizes_ha.size else np.nan,
                "median_field_ha": float(np.median(sizes_ha)) if sizes_ha.size else np.nan,
                # area-weighted mean: field size experienced by a random point in cropland
                "aw_mean_field_ha": float((sizes_ha**2).sum() / sizes_ha.sum())
                if sizes_ha.size else np.nan,
                "mean_confidence": float(sub["confidence"].mean())
                if "confidence" in sub else np.nan,
                # share of fields with no confidence value, for sensitivity checks
                "frac_conf_null": float(sub["confidence"].isna().mean())
                if "confidence" in sub else np.nan,
            }
            rows.append(out)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pull")
    a.add_argument("--year", type=int, default=2025)
    a.add_argument("--out", required=True)
    a.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                   help="lon/lat bounding box (default: NY-PA)")
    b = sub.add_parser("metrics")
    b.add_argument("--fields", required=True)
    b.add_argument("--sites", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--radii", type=int, nargs="+", default=RADII_M)
    args = p.parse_args()

    if args.cmd == "pull":
        pull(args.year, args.out, bbox=tuple(args.bbox) if args.bbox else NYPA_BBOX)
    else:
        fields = load_fields(args.fields)
        s = pd.read_csv(args.sites)
        sites = gpd.GeoDataFrame(
            s, geometry=gpd.points_from_xy(s.lon, s.lat), crs="EPSG:4326"
        ).to_crs(EQUAL_AREA)
        site_metrics(fields, sites, args.radii).to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
