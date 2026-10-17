"""Generate a small synthetic dataset with the exact schema of the DrivenData files.
Used ONLY to smoke-test the pipeline; never report results from it."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SUP = ["adobe_mud", "mud_mortar_stone", "stone_flag", "cement_mortar_stone", "mud_mortar_brick",
       "cement_mortar_brick", "timber", "bamboo", "rc_non_engineered", "rc_engineered", "other"]
SEC = ["agriculture", "hotel", "rental", "institution", "school", "industry",
       "health_post", "gov_office", "use_police", "other"]
CATS = {"land_surface_condition": "not", "foundation_type": "hirwu", "roof_type": "nqx",
        "ground_floor_type": "fmvxz", "other_floor_type": "jqsx", "position": "jost",
        "plan_configuration": "acdfmnoqsu", "legal_ownership_status": "arvw"}


def make(n, rng):
    d = {"geo_level_1_id": rng.integers(0, 31, n), "geo_level_2_id": rng.integers(0, 1400, n),
         "geo_level_3_id": rng.integers(0, 12000, n), "count_floors_pre_eq": rng.integers(1, 6, n),
         "age": rng.choice([0, 5, 10, 15, 20, 30, 50, 100], n),
         "area_percentage": rng.integers(2, 20, n), "height_percentage": rng.integers(2, 10, n)}
    for c, v in CATS.items():
        d[c] = rng.choice(list(v), n)
    for s in SUP:
        d["has_superstructure_" + s] = (rng.random(n) < 0.1).astype(int)
    d["count_families"] = rng.integers(0, 4, n)
    d["has_secondary_use"] = (rng.random(n) < 0.1).astype(int)
    for s in SEC:
        d["has_secondary_use_" + s] = (rng.random(n) < 0.03).astype(int)
    df = pd.DataFrame(d)
    score = (0.5 * df.count_floors_pre_eq + 0.02 * df.age + 1.5 * df.has_superstructure_mud_mortar_stone
             - 1.5 * df.has_superstructure_rc_engineered + 0.05 * df.geo_level_1_id / 3
             + (df.foundation_type == "r") * -1.0 + rng.normal(0, 1, n))
    y = np.digitize(score, np.quantile(score, [0.15, 0.7])) + 1
    df.index = pd.Index(rng.choice(10**6, n, replace=False), name="building_id")
    return df, pd.Series(y, index=df.index, name="damage_grade")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data_synth"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True); rng = np.random.default_rng(0)
    X, y = make(6000, rng); X.to_csv(out / "train_values.csv"); y.to_csv(out / "train_labels.csv")
    Xt, _ = make(2000, rng); Xt.to_csv(out / "test_values.csv")
    print("wrote", out)
