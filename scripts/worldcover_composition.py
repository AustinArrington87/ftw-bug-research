"""
Land-cover composition around survey cells from ESA WorldCover 10 m 2021 v200
(Zanaga et al. 2022; doi:10.5281/zenodo.7254221; CC-BY 4.0), for the
peer-review revision (semi-natural habitat / residual composition).

For each site (cell centre) and radius, writes the fraction of valid pixels in
each class: tree (10), shrub (20), grass (30), crop (40), built (50),
bare (60), water (80), wetland (90). Semi-natural = tree+shrub+grass+wetland.

Usage (run where sites.csv and sites_cb.csv are; needs ~4-6 GB disk):
    pip install rasterio numpy pandas requests
    python worldcover_composition.py sites.csv sites_cb.csv --out worldcover_composition.csv
Upload worldcover_composition.csv (a few MB).
"""
import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import requests

BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
PX = 1 / 12000            # 0.0000833 deg, 10 m at the equator
TILE = 3                  # 3 x 3 degree tiles = 36000 px
CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
           60: "bare", 80: "water", 90: "wetland"}
RADII = [1000, 2000, 3000]


def tile_name(lat0, lon0):
    ns = "N" if lat0 >= 0 else "S"; ew = "E" if lon0 >= 0 else "W"
    return f"ESA_WorldCover_10m_2021_v200_{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}_Map.tif"


def needed_tiles(lat, lon, pad=0.05):
    lats = range(int(math.floor((lat.min() - pad) / 3) * 3), int(math.floor((lat.max() + pad) / 3) * 3) + 1, 3)
    lons = range(int(math.floor((lon.min() - pad) / 3) * 3), int(math.floor((lon.max() + pad) / 3) * 3) + 1, 3)
    return [(a, b) for a in lats for b in lons]


def download(tiles, folder):
    folder.mkdir(exist_ok=True); have = []
    for a, b in tiles:
        name = tile_name(a, b); dest = folder / name
        if not dest.exists():
            r = requests.get(BASE + name, stream=True, timeout=120)
            if r.status_code == 404:          # tile entirely water / no data
                print("no tile (skipped):", name); continue
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 22): f.write(chunk)
            os.replace(tmp, dest); print("downloaded", name)
        have.append((a, b, dest))
    return have


def write_vrt(have, path):
    west = min(b for _, b, _ in have); east = max(b for _, b, _ in have) + TILE
    south = min(a for a, _, _ in have); north = max(a for a, _, _ in have) + TILE
    W = round((east - west) / PX); H = round((north - south) / PX); n = round(TILE / PX)
    src = []
    for a, b, f in have:
        xo = round((b - west) / PX); yo = round((north - (a + TILE)) / PX)
        src.append(f"""<SimpleSource><SourceFilename relativeToVRT="0">{f.resolve()}</SourceFilename>
<SourceBand>1</SourceBand><SrcRect xOff="0" yOff="0" xSize="{n}" ySize="{n}"/>
<DstRect xOff="{xo}" yOff="{yo}" xSize="{n}" ySize="{n}"/></SimpleSource>""")
    path.write_text(f"""<VRTDataset rasterXSize="{W}" rasterYSize="{H}">
<SRS>EPSG:4326</SRS><GeoTransform>{west}, {PX}, 0, {north}, 0, {-PX}</GeoTransform>
<VRTRasterBand dataType="Byte" band="1"><NoDataValue>0</NoDataValue>
{''.join(src)}
</VRTRasterBand></VRTDataset>""")


def composition(ds, lat, lon, radii):
    rmax = max(radii)
    mlat = 111320.0; mlon = 111320.0 * math.cos(math.radians(lat))
    dlat, dlon = rmax / mlat, rmax / mlon
    row0, col0 = ds.index(lon - dlon, lat + dlat)
    row1, col1 = ds.index(lon + dlon, lat - dlat)
    win = Window(col0, row0, col1 - col0 + 1, row1 - row0 + 1)
    a = ds.read(1, window=win, boundless=True, fill_value=0)
    t = ds.window_transform(win)
    cols = np.arange(a.shape[1]) + 0.5; rows = np.arange(a.shape[0]) + 0.5
    xs = t.c + cols * t.a; ys = t.f + rows * t.e
    dx = (xs - lon) * mlon; dy = (ys - lat) * mlat
    d2 = dy[:, None] ** 2 + dx[None, :] ** 2
    out = []
    for r in radii:
        m = (d2 <= r * r) & (a > 0)
        vals = a[m]; n = vals.size
        row = {"radius_m": r, "valid_px": int(n)}
        for code, name in CLASSES.items():
            row[f"wc_{name}"] = float((vals == code).sum() / n) if n else np.nan
        out.append(row)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sites", nargs="+"); p.add_argument("--out", default="worldcover_composition.csv")
    p.add_argument("--tiles", default="worldcover_tiles")
    a = p.parse_args()
    s = pd.concat([pd.read_csv(f)[["site_id", "lat", "lon"]] for f in a.sites]).drop_duplicates("site_id")
    have = download(needed_tiles(s.lat, s.lon), Path(a.tiles))
    vrt = Path(a.tiles) / "mosaic.vrt"; write_vrt(have, vrt)
    rows = []
    with rasterio.open(vrt) as ds:
        for i, r in enumerate(s.itertuples(index=False)):
            for c in composition(ds, r.lat, r.lon, RADII):
                rows.append({"site_id": r.site_id, **c})
            if i % 2000 == 0: print(f"{i}/{len(s)} sites", flush=True)
    out = pd.DataFrame(rows)
    out["wc_seminatural"] = out[["wc_tree", "wc_shrub", "wc_grass", "wc_wetland"]].sum(axis=1)
    out.to_csv(a.out, index=False); print("wrote", a.out, len(out))


if __name__ == "__main__":
    main()
