<p align="center">
  <img src="assets/svg/pipeline.svg" width="100%" alt="Animated pipeline: training lane and inference lane of the Game Launch IDSS">
</p>

# Game Launch IDSS: guided documentation

This folder explains, in a fixed order, every algorithm and technology the project uses. Each chapter follows the same pattern:

1. **the idea** in plain words,
2. **an animation** drawn from the real model output (not a mock-up),
3. **the mathematics**, when there is any,
4. **where it lives in the code**, with file and function names,
5. **a link to the matching step of the Interactive Lab**, where you can change inputs and watch the computation react.

> **Two ways to explore.** The chapters below are Markdown with animated SVG figures, readable directly on GitHub.
> The **[Interactive Lab](https://cocopops9.github.io/Game_Launch_IDSS/)** (`docs/index.html`) runs the project's exported XGBoost model in your browser,
> so every slider and checkbox recomputes the real percentile. GitHub does not execute JavaScript inside Markdown, which is why the
> interactive part is a separate page. See [Opening the Interactive Lab](#opening-the-interactive-lab) below.

## Reading path

| # | Chapter | You will learn | Lab step |
|---|---------|----------------|----------|
| 1 | [Overview](01_overview.md) | the decision problem, the three models, why the answer is a percentile | [1](https://cocopops9.github.io/Game_Launch_IDSS/#intro) |
| 2 | [Architecture and technology stack](02_architecture_and_stack.md) | modules, Streamlit caching, session state, JSON persistence | [1](https://cocopops9.github.io/Game_Launch_IDSS/#intro) |
| 3 | [The dataset](03_dataset.md) | `steam.csv`, owners ranges, censoring, skew, the two splits | [2](https://cocopops9.github.io/Game_Launch_IDSS/#data) |
| 4 | [Feature engineering](04_feature_engineering.md) | the 91 ranker features, the form mapping, hidden defaults | [3, 4](https://cocopops9.github.io/Game_Launch_IDSS/#features) |
| 5 | [Gradient boosting with XGBoost](05_gradient_boosting.md) | additive trees, the regularized objective, the hyperparameters | [5](https://cocopops9.github.io/Game_Launch_IDSS/#boosting) |
| 6 | [Rank target and Spearman correlation](06_rank_target_and_spearman.md) | why the model learns ranks, how ties behave, how to read 0.725 | [6](https://cocopops9.github.io/Game_Launch_IDSS/#ranking) |
| 7 | [From score to percentile](07_percentile_mapping.md) | the reference distribution, `searchsorted`, clipping | [7](https://cocopops9.github.io/Game_Launch_IDSS/#percentile) |
| 8 | [The what-if engine](08_what_if_engine.md) | 13 counterfactual changes, dependencies, confidence labels | [8](https://cocopops9.github.io/Game_Launch_IDSS/#whatif) |
| 9 | [The decision layer](09_decision_layer.md) | tiers, risk rules, readiness score, insights | [9](https://cocopops9.github.io/Game_Launch_IDSS/#dashboard) |
| 10 | [Secondary models and validation](10_secondary_models_and_validation.md) | owners regression, review regression, every reported metric | [10](https://cocopops9.github.io/Game_Launch_IDSS/#validation) |
| 11 | [Limitations and implementation notes](11_limitations.md) | what the numbers do not mean, verified quirks of the code | [10](https://cocopops9.github.io/Game_Launch_IDSS/#validation) |

## Opening the Interactive Lab

The lab is a static page with no build step and no server code.

* **On GitHub Pages:** in the repository settings, open *Pages*, choose *Deploy from a branch*, select `main` and the `/docs` folder, and save.
  The lab is then served at `https://cocopops9.github.io/Game_Launch_IDSS/`. The file `docs/.nojekyll` tells Pages to serve the files as they are.
* **Locally, without a server:** open `docs/index.html` in a browser. The data files are plain JavaScript, so this works from `file://`.
* **Locally, with a server:** `python -m http.server 8000 --directory docs`, then visit `http://localhost:8000`.

## Where the lab's numbers come from

`docs/tools/export_lab_data.py` trains the ranker exactly like the app does (`SteamDataLoader` followed by `IntelligentRanker.train`)
and writes two files:

| File | Content |
|------|---------|
| `assets/data/ranker_model.js` | the 150 XGBoost trees, the 91 feature names, the base score, the 5,415 sorted reference scores, the validation results, 25 test rows with their Python predictions |
| `assets/data/dataset_stats.js` | dataset distributions, 800 held-out games for the scatter plot, one sample game per owners range |

`assets/js/idss-core.js` is a function-by-function port of `_build_ranker_features`, `_predict_rank`, `get_percentile`,
`get_improvements` and `analyze_game`. On load, the lab re-evaluates the 25 stored test rows and prints the largest difference from
Python's `model.predict`; for the committed snapshot it is 2.5e-7 (float32 rounding). Three full configurations were also checked
against the Python engine and produced identical percentiles, gains and confidence labels.

After retraining or changing the model code, regenerate the snapshot from the repository root:

```bash
python docs/tools/export_lab_data.py
```

The animated SVG figures embed numbers from the snapshot of **2026-10-22** (XGBoost 3.4.1, pandas 2.3.3).
