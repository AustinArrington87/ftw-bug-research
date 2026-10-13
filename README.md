# Field size, edge density, and pollinator reporting rates across a US cropland gradient

Code and derived data for:

> Arrington, A. B. (submitted). Field size and edge density do not predict bee and hoverfly reporting rates across a cropland gradient in six US states: a test with AI-derived field boundaries. *Frontiers in Sustainable Food Systems* (Agroecology and Ecosystem Services).

## Data sources
- **Field boundaries:** Fields of The World Global 2024–2025 (Robinson et al., 2026; CC-BY-4.0), https://source.coop/ftw/global-data
- **Pollinator records:** research-grade iNaturalist bees (Anthophila) and hoverflies (Syrphidae), 2017–2025, in NY, PA, OH, IN, IL, IA, plus the Empire State Native Pollinator Survey iNaturalist project. NY–PA observation IDs: `data/derived/observation_ids.csv`.

Observer usernames are replaced with anonymous IDs (`obs00000`…), using one mapping across all six states, so analyses that use observer identity reproduce exactly.

## Layout
- `data/derived/`: NY–PA sites, visits, detections, landscape metrics (`landscape_avg.csv`, `landscape_avg_nosliver.csv`, per-year `site_metrics_*`)
- `data/derived/cornbelt/`: IA, IL, IN, OH equivalents (`*_cb*`)
- `results/nypa/`: primary models, replication, declared species analysis, robustness
- `results/extension/`: declared cropland-gradient extension (+ other-bee 1-km sensitivity)
- `figures/`: Figures 1–4 and S1 (TIFF 300 DPI and PNG)
- `scripts/`: full pipeline

## Pipeline
| Step | Script |
|---|---|
| Pull FTW polygons (DuckDB, anonymous S3, UTC timestamps, `--bbox`) | `ftw_landscape_metrics.py pull` |
| Pull iNaturalist records (`--bbox`/`--place-id`) | `inat_pollinator_sites.py pull` |
| Build 1-km sites, visits (observer × date), detections; clip to states | `inat_pollinator_sites.py build` |
| Landscape metrics at 1/2/3 km (5 m simplification; shared edges once; `--min-area-m2` sliver check) | `ftw_metrics_lean.py` |
| Corn Belt end-to-end driver | `run_cornbelt.sh` |
| Primary reporting-rate models (NY–PA) | `reporting_rate_models.py` |
| Declared species analysis | `secondary_bombus_species.py` |
| Robustness checks | `robustness_checks.py` |
| Declared cropland-gradient extension | `cropland_gradient_extension.py` |
| Figures | `make_figures.py`, `make_figures_extension.py` |

`occupancy_models.py` is the originally planned occupancy analysis, retained for transparency; it was not identifiable in these data (see Methods).

### Reproduce the results from the included data
```bash
pip install -r requirements.txt
cd scripts
python reporting_rate_models.py ../data/derived ../results/nypa
python reporting_rate_models.py ../data/derived ../results/nypa --years 2017 2018 2019 2020 2021 2022 2023
python secondary_bombus_species.py ../data/derived ../results/nypa
python robustness_checks.py ../data/derived ../results/nypa
python cropland_gradient_extension.py ../data/derived ../data/derived/cornbelt ../results/extension
```

## Inclusion rule (important)
A survey cell is analyzed only if it has ≥5% FTW cropland within **each** of the 1-, 2-, and 3-km radii. Edge per cropland hectare is a ratio and becomes unstable where cropland in the buffer is tiny (clipped slivers produced values >400,000 m/ha). Cells with computed cropland > 1 (overlapping FTW polygons; 3 Corn Belt cells) are excluded.

## Notes for FTW users
- Read `determination:datetime` in UTC; local US time shifts year labels back by one.
- Set `s3_url_style='path'` in DuckDB (the bucket name contains dots).
- This release had no per-field confidence values for these states.
- Simplifying 10 m polygons at 5 m removes stair-stepping that inflates perimeters by ~6%.
