"""
Memory-lean landscape metrics from FTW field polygons, for thousands of sites.

Same metrics as ftw_landscape_metrics.py, plus edge length per hectare of
cropland (a configuration measure less tied to cropland amount). Geometry is
projected to EPSG:5070 and simplified at 5 m (half a Sentinel-2 pixel) to
remove raster stair-stepping, which otherwise inflates perimeters.

  python ftw_metrics_lean.py FIELDS.parquet SITES.csv OUT.csv [--radii 1000 2000 3000]
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

SIMPLIFY_M = 5.0
TR = Transformer.from_crs(4326, 5070, always_xy=True)


def _proj(coords):
    x, y = TR.transform(coords[:, 0], coords[:, 1])
    return np.column_stack([x, y])


def load(path, min_area_m2=0.0):
    pf = pq.ParquetFile(path)
    geoms = []
    for i in range(pf.num_row_groups):
        wkb = pf.read_row_group(i, columns=["geometry"]).column(0)
        g = shapely.from_wkb(wkb.to_numpy(zero_copy_only=False))
        g = shapely.simplify(shapely.transform(g, _proj), SIMPLIFY_M)
        geoms.append(g[~shapely.is_empty(g)])
    g = np.concatenate(geoms)
    g = shapely.make_valid(g)
    if min_area_m2 > 0:
        g = g[shapely.area(g) >= min_area_m2]
    return g


def metrics(fields, sites_xy, site_ids, radii):
    tree = shapely.STRtree(fields)
    area = shapely.area(fields)
    rep = shapely.point_on_surface(fields)
    rows = []
    for sid, (x, y) in zip(site_ids, sites_xy):
        pt = shapely.Point(x, y)
        for r in radii:
            buf = pt.buffer(r, quad_segs=16)
            buf_ha = buf.area / 1e4
            idx = tree.query(buf, predicate="intersects")
            row = {"site_id": sid, "radius_m": r}
            if idx.size == 0:
                rows.append(row | {"n_fields": 0, "cropland_frac": 0.0,
                                   "edge_density_m_per_ha": 0.0})
                continue
            crop_ha = shapely.area(shapely.intersection(fields[idx], buf)).sum() / 1e4
            edges = shapely.intersection(shapely.boundary(fields[idx]), buf)
            edge_m = shapely.union_all(edges).length  # shared edges once
            inside = shapely.within(rep[idx], buf)
            sizes = area[idx][inside] / 1e4
            row |= {
                "n_fields": int(inside.sum()),
                "cropland_frac": crop_ha / buf_ha,
                "edge_density_m_per_ha": edge_m / buf_ha,
                "edge_per_crop_ha": edge_m / crop_ha if crop_ha > 0 else np.nan,
                "mean_field_ha": sizes.mean() if sizes.size else np.nan,
                "median_field_ha": float(np.median(sizes)) if sizes.size else np.nan,
                "aw_mean_field_ha": (sizes**2).sum() / sizes.sum() if sizes.size else np.nan,
                "frac_fields_lt_0p1ha": (sizes < 0.1).mean() if sizes.size else np.nan,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fields"); p.add_argument("sites"); p.add_argument("out")
    p.add_argument("--radii", type=int, nargs="+", default=[1000, 2000, 3000])
    p.add_argument("--state", help="only sites whose 'state' column equals this")
    p.add_argument("--min-area-m2", type=float, default=0.0,
                   help="drop fields smaller than this (1000 = 0.1 ha sliver check)")
    a = p.parse_args()
    fields = load(a.fields, a.min_area_m2)
    s = pd.read_csv(a.sites)
    if a.state:
        s = s[s.state == a.state]
    x, y = TR.transform(s.lon.to_numpy(), s.lat.to_numpy())
    m = metrics(fields, np.column_stack([x, y]), s.site_id, a.radii)
    m.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
