#!/usr/bin/env bash
# Declared extension: Iowa, Illinois, Indiana, Ohio (see Methods, "Declared
# extension: the cropland gradient"). Same sources, filters, and metrics as NY-PA.
#
# Run from the folder that holds the updated scripts:
#   ftw_landscape_metrics.py  inat_pollinator_sites.py  ftw_metrics_lean.py
# Usage:  bash run_cornbelt.sh
# Output: cornbelt_upload.zip  (upload this; nothing else is needed)
#
# Time: iNat pulls dominate (sequential, ~1 request/sec, likely 1-3 h total).
# Memory: metrics load one state-year of FTW polygons at a time; Illinois/Iowa
#         may need ~6-10 GB RAM.
set -euo pipefail

WORK=cornbelt
mkdir -p "$WORK"/{raw,ftw,logs,metrics,upload}
D1=2017-01-01; D2=2025-12-31
BEES=630955; SYRPHIDS=49995

# State bounding boxes (W S E N), buffered ~0.05 deg so fields just across a
# border are still counted for sites near it. Census STATEFP codes for clipping.
bbox() {  # portable (macOS ships bash 3.2, no associative arrays)
  case "$1" in
    IA) echo "-96.69 40.33 -90.09 43.55" ;;
    IL) echo "-91.56 36.92 -87.44 42.56" ;;
    IN) echo "-88.15 37.72 -84.73 41.81" ;;
    OH) echo "-84.87 38.35 -80.47 42.03" ;;
  esac
}
STATES="IA IL IN OH"
PY="${PY:-python3}"   # override with: PY=python bash run_cornbelt.sh

# ---- 1. State boundaries (Census cartographic, 1:500k) --------------------
if [ ! -f "$WORK/cornbelt_states.gpkg" ]; then
  ZIP=tiger_download/cb_2023_us_state_500k.zip
  mkdir -p tiger_download
  [ -f "$ZIP" ] || curl -sSL -o "$ZIP" \
    https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip
  "$PY" -u - <<'EOF'
import geopandas as gpd
g = gpd.read_file("zip://tiger_download/cb_2023_us_state_500k.zip")
g = g[g.STATEFP.isin(["19", "17", "18", "39"])][["STATEFP", "STUSPS", "NAME", "geometry"]]
assert sorted(g.STUSPS) == ["IA", "IL", "IN", "OH"], g.STUSPS.tolist()
g.to_file("cornbelt/cornbelt_states.gpkg", layer="states", driver="GPKG")
print("boundaries:", g.STUSPS.tolist())
EOF
fi

# ---- 2. FTW pulls in the background (S3, no rate limit) ------------------
(
  for ST in $STATES; do
    for YR in 2024 2025; do
      OUT="$WORK/ftw/ftw_${ST}_${YR}.parquet"
      [ -f "$OUT" ] && continue
      "$PY" -u ftw_landscape_metrics.py pull --year "$YR" --bbox $(bbox "$ST") \
        --out "$OUT" > "$WORK/logs/ftw_${ST}_${YR}.log" 2>&1
    done
  done
) &
FTW_PID=$!

# ---- 3. iNaturalist pulls, strictly one at a time (rate limit) -----------
for ST in $STATES; do
  for TX in bees syrphids; do
    OUT="$WORK/raw/${TX}_${ST}.csv"
    [ -f "$OUT" ] && continue
    TID=$BEES; [ "$TX" = syrphids ] && TID=$SYRPHIDS
    "$PY" -u inat_pollinator_sites.py pull --project "" --taxon-id "$TID" \
      --bbox $(bbox "$ST") --d1 $D1 --d2 $D2 --out "$OUT" \
      > "$WORK/logs/inat_${TX}_${ST}.log" 2>&1
    echo "done: $OUT"
  done
done

echo "waiting for FTW pulls..."; wait $FTW_PID

# ---- 4. Merge, de-duplicate (bboxes overlap at borders), build ----------
"$PY" -u - <<'EOF'
import glob, pandas as pd
parts = [pd.read_csv(f).assign(source="inat_" + f.split("/")[-1].split("_")[0])
         for f in sorted(glob.glob("cornbelt/raw/*.csv"))]
a = pd.concat(parts, ignore_index=True).drop_duplicates("obs_id")
a.to_csv("cornbelt/obs_cornbelt.csv", index=False)
print("unique records:", len(a))
EOF
"$PY" -u inat_pollinator_sites.py build --obs "$WORK/obs_cornbelt.csv" --cell-m 1000 \
  --boundary "$WORK/cornbelt_states.gpkg" \
  --sites "$WORK/upload/sites_cb.csv" --visits "$WORK/upload/visits_cb.csv" \
  --detections "$WORK/upload/detections_cb.csv"

# ---- 5. Landscape metrics per state-year (+ sliver check) ----------------
for ST in $STATES; do
  for YR in 2024 2025; do
    F="$WORK/ftw/ftw_${ST}_${YR}.parquet"
    "$PY" -u ftw_metrics_lean.py "$F" "$WORK/upload/sites_cb.csv" \
      "$WORK/metrics/m_${ST}_${YR}.csv" --state "$ST"
    "$PY" -u ftw_metrics_lean.py "$F" "$WORK/upload/sites_cb.csv" \
      "$WORK/metrics/ns_${ST}_${YR}.csv" --state "$ST" --min-area-m2 1000
  done
done
"$PY" -u - <<'EOF'
import glob, pandas as pd
for kind, name in [("m", "site_metrics"), ("ns", "site_metrics_nosliver")]:
    for yr in (2024, 2025):
        fs = sorted(glob.glob(f"cornbelt/metrics/{kind}_*_{yr}.csv"))
        pd.concat([pd.read_csv(f) for f in fs]).to_csv(
            f"cornbelt/upload/{name}_cb_{yr}.csv", index=False)
print("metrics combined")
EOF

# ---- 6. Package -----------------------------------------------------------
cp "$WORK"/logs/*.log "$WORK/upload/" 2>/dev/null || true
(cd "$WORK" && rm -f ../cornbelt_upload.zip && zip -qr ../cornbelt_upload.zip upload)
echo "DONE -> cornbelt_upload.zip"
