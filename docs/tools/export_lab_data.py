"""
Export Lab Data for the Interactive Documentation
=================================================

This script trains the project's ranking model exactly as the Streamlit app
does (SteamDataLoader -> IntelligentRanker) and exports a browser-friendly
snapshot of it, so that the interactive pages in docs/ can run the real model
client-side, without a Python server.

Outputs (written to docs/assets/data/):
    ranker_model.js   - XGBoost trees, feature list, reference distribution,
                        validated improvements, metrics
    dataset_stats.js  - dataset distributions and held-out sample games

Usage (from the repository root):
    python docs/tools/export_lab_data.py

The output files are plain JavaScript (window.IDSS_* = {...}) instead of JSON
so that docs/index.html also works when opened directly from disk (file://),
where fetch() of local JSON files is blocked by browsers.
"""

import os
import sys
import json
from datetime import date

import numpy as np
import pandas as pd
import xgboost
from scipy.stats import rankdata

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from data_loader import SteamDataLoader          # noqa: E402
from ranking_integration import IntelligentRanker  # noqa: E402

OUTPUT_DIR = os.path.join(REPO_ROOT, "docs", "assets", "data")

OWNER_TIER_LABELS = {
    10000: "0 - 20k", 35000: "20k - 50k", 75000: "50k - 100k",
    150000: "100k - 200k", 350000: "200k - 500k", 750000: "500k - 1M",
    1500000: "1M - 2M", 3500000: "2M - 5M", 7500000: "5M - 10M",
    15000000: "10M - 20M", 35000000: "20M - 50M", 75000000: "50M - 100M",
    150000000: "100M - 200M",
}


def round_sig(value, digits=7):
    """Round a float to a fixed number of significant digits (float32 precision)."""
    return float(f"{float(value):.{digits}g}")


def train_ranker_like_the_app():
    """Reproduce the app path: the loader preprocesses, then the ranker trains on it."""
    csv_path = os.path.join(REPO_ROOT, "steam.csv")
    loader = SteamDataLoader(steam_csv_path=csv_path)
    df = loader.load_steam_data()

    ranker = IntelligentRanker(df)
    ranker.train()
    return ranker


def read_base_score(booster):
    """Read the global bias that XGBoost adds to the sum of all tree leaves."""
    config = json.loads(booster.save_config())
    raw = config["learner"]["learner_model_param"]["base_score"]
    # XGBoost 2.x stores "5E-1", XGBoost 3.x stores "[5E-1]"
    return float(str(raw).strip("[]"))


def flatten_tree(tree_json, feature_index):
    """
    Convert one nested XGBoost JSON tree into four parallel arrays:
        f[i]  feature index of the split (-1 for a leaf)
        t[i]  split threshold, or the leaf value when f[i] == -1
        y[i]  position of the child taken when x < threshold
        n[i]  position of the child taken otherwise
    Missing values never occur here, because the app fills them with 0.
    """
    nodes = {}

    def visit(node):
        nodes[node["nodeid"]] = node
        for child in node.get("children", []):
            visit(child)

    visit(tree_json)
    order = sorted(nodes)
    position = {node_id: i for i, node_id in enumerate(order)}

    f, t, y, n = [], [], [], []
    for node_id in order:
        node = nodes[node_id]
        if "leaf" in node:
            f.append(-1)
            t.append(round_sig(node["leaf"]))
            y.append(-1)
            n.append(-1)
        else:
            f.append(feature_index[node["split"]])
            t.append(round_sig(node["split_condition"]))
            y.append(position[node["yes"]])
            n.append(position[node["no"]])

    root = position[tree_json["nodeid"]]
    return {"root": root, "f": f, "t": t, "y": y, "n": n}


def export_model(ranker):
    model = ranker.ranking_models["xgb_rank"]
    booster = model.get_booster()
    feature_index = {name: i for i, name in enumerate(ranker.feature_cols)}

    trees = [flatten_tree(json.loads(dump), feature_index)
             for dump in booster.get_dump(dump_format="json")]

    importance = sorted(zip(ranker.feature_cols, model.feature_importances_),
                        key=lambda item: item[1], reverse=True)

    validated = {}
    for feature, info in ranker.validated_improvements.items():
        validated[feature] = {
            "name": info["name"],
            "confidence": info["confidence"],
            "actual_lift": round(float(info["actual_lift"]), 1),
            "n_with": int(info["n_with"]),
            "n_without": int(info["n_without"]),
        }

    # Real test-set rows, used by the docs to verify the JavaScript evaluator
    check_rows = ranker.X_test.head(25).fillna(0).replace([np.inf, -np.inf], 0)
    check_predictions = model.predict(check_rows)

    return {
        "generated": date.today().isoformat(),
        "xgboost_version": xgboost.__version__,
        "pandas_version": pd.__version__,
        "spearman": round(float(ranker.spearman_score), 4),
        "n_train": int(len(ranker.X_train)),
        "n_test": int(len(ranker.X_test)),
        "hyperparameters": {
            key: model.get_params()[key]
            for key in ["n_estimators", "learning_rate", "max_depth",
                        "min_child_weight", "subsample", "colsample_bytree", "random_state"]
        },
        "feature_cols": list(ranker.feature_cols),
        "base_score": read_base_score(booster),
        "trees": trees,
        "reference_ranks": [round_sig(v) for v in np.sort(ranker.reference_ranks)],
        "validated_improvements": validated,
        "feature_importance": [[name, round(float(value), 5)] for name, value in importance],
        "check_rows": [[round_sig(v) for v in row] for row in check_rows.values],
        "check_predictions": [round_sig(v) for v in check_predictions],
    }


def export_dataset_stats(ranker):
    df = ranker.df
    owners = df["owners"]

    owners_tiers = [
        {"midpoint": int(mid), "label": OWNER_TIER_LABELS.get(int(mid), str(int(mid))), "count": int(count)}
        for mid, count in owners.value_counts().sort_index().items()
    ]

    release_years = df["release_date"].dt.year.value_counts().sort_index()
    price_tiers = df["price_tier"].value_counts().sort_index()

    # Held-out (test-set) sample: the most reviewed game of every owners tier
    model = ranker.ranking_models["xgb_rank"]
    test = df.loc[ranker.X_test.index].copy()
    test["predicted_rank"] = model.predict(ranker.X_test)
    sorted_reference = np.sort(ranker.reference_ranks)
    test["predicted_percentile"] = (np.searchsorted(sorted_reference, test["predicted_rank"])
                                    / len(sorted_reference) * 100).clip(1, 99)
    test["actual_percentile"] = rankdata(test["owners"]) / len(test) * 100
    test["reviews"] = (pd.to_numeric(test["positive_ratings"], errors="coerce").fillna(0)
                       + pd.to_numeric(test["negative_ratings"], errors="coerce").fillna(0))

    samples = []
    for midpoint, group in test.groupby("owners"):
        best = group.sort_values("reviews", ascending=False).iloc[0]
        samples.append({
            "name": str(best["name"]),
            "owners": OWNER_TIER_LABELS.get(int(midpoint), str(int(midpoint))),
            "release_year": int(best["release_year"]),
            "price": float(best["price"]),
            "predicted_percentile": round(float(best["predicted_percentile"]), 1),
            "actual_percentile": round(float(best["actual_percentile"]), 1),
        })

    # Scatter sample: predicted score against actual owners tier for 800 held-out games
    scatter = test.sample(n=min(800, len(test)), random_state=7)
    tier_order = {int(mid): i for i, mid in enumerate(sorted(owners.unique()))}
    scatter_points = [[tier_order[int(o)], round_sig(p, 4)]
                      for o, p in zip(scatter["owners"], scatter["predicted_rank"])]

    return {
        "n_games": int(len(df)),
        "owners_tiers": owners_tiers,
        "release_years": {int(k): int(v) for k, v in release_years.items()},
        "price_tiers": {int(k): int(v) for k, v in price_tiers.items()},
        "platform_count": {int(k): int(v) for k, v in df["platform_count"].value_counts().sort_index().items()},
        "share_free": round(float(df["is_free"].mean()), 4),
        "share_mac": round(float(df["mac"].mean()), 4),
        "share_linux": round(float(df["linux"].mean()), 4),
        "share_achievements": round(float(df["has_achievements"].mean()), 4),
        "share_age_clipped_to_zero": round(float((df["game_age_days"] == 0).mean()), 4),
        "sample_games": samples,
        "scatter": scatter_points,
    }


def write_js(filename, variable, payload):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        f.write("// Generated by docs/tools/export_lab_data.py. Do not edit by hand.\n")
        f.write(f"window.{variable} = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")
    print(f"Wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def main():
    ranker = train_ranker_like_the_app()
    write_js("ranker_model.js", "IDSS_MODEL", export_model(ranker))
    write_js("dataset_stats.js", "IDSS_DATA", export_dataset_stats(ranker))


if __name__ == "__main__":
    main()
