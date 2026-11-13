#!/usr/bin/env bash
# Peer-review revision data pull. Run from the same folder as run_cornbelt.sh
# (it needs cornbelt/obs_cornbelt.csv, cornbelt/ftw/*.parquet, cornbelt/cornbelt_states.gpkg).
# Put these files next to it first: worldcover_composition.py, cornbelt_precise.py,
# sites_all_six_states.csv, and the current ftw_metrics_lean.py + inat_pollinator_sites.py.
# Output: revision_upload.zip  (upload only this)
set -euo pipefail
PY="${PY:-python3}"
"$PY" -m pip install --quiet rasterio requests || true
mkdir -p cornbelt/upload_rev
# 1. Land-cover composition (ESA WorldCover 2021) around all 34,939 cells, 1/2/3 km
"$PY" -u worldcover_composition.py sites_all_six_states.csv --out cornbelt/upload_rev/worldcover_composition.csv
# 2. Corn Belt record-level table + FTW metrics at record coordinates (<=100 m accuracy)
"$PY" -u cornbelt_precise.py
(cd cornbelt && rm -f ../revision_upload.zip && zip -qr ../revision_upload.zip upload_rev)
echo "DONE -> revision_upload.zip"
