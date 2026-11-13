"""Create the working layout the review scripts expect, from the repository data.
Run from scripts/:  python setup_review_layout.py
Creates data_v3/, results_ext/data_pooled/, data_obs/ next to the scripts."""
import shutil
from pathlib import Path
import cropland_gradient_extension as E
D = Path("../data/derived")
Path("data_v3").mkdir(exist_ok=True)
for f in ["sites.csv", "visits.csv", "detections.csv", "landscape_avg.csv", "landscape_avg_nosliver.csv", "worldcover_composition.csv"]:
    shutil.copy(D / f, Path("data_v3") / f)
pooled = E.build_pooled(D, D / "cornbelt", Path("results_ext/data_pooled"))
shutil.copy(D / "worldcover_composition.csv", pooled / "worldcover_composition.csv")
shutil.copytree(D / "obs", "data_obs", dirs_exist_ok=True)
print("layout ready")
