[Previous: Secondary models and validation](10_secondary_models_and_validation.md) · [Index](README.md) · **11 Limitations**

# 11. Limitations and implementation notes

> **Interactive:** [Lab step 10](https://cocopops9.github.io/Game_Launch_IDSS/#validation)

An honest decision support system states where its numbers stop being reliable. This chapter separates
**scientific limitations** (properties of the data and the method) from **implementation notes** (specific behaviors of the
current code, each verified by running it). None of the notes below were changed in the code by this documentation;
they describe the repository as it is.

## 11.1 Scientific limitations

| # | Limitation | Evidence | Practical meaning |
|---|---|---|---|
| S1 | **Association, not causation.** The model learns which features co-occur with high owners. | Trading Cards are enabled by Valve after a sales threshold (since 2017); tags describe content, they do not create it. | Treat a what-if gain as "games like this ranked higher", not "this change will add owners". |
| S2 | **Optimistic validation.** The headline 0.725 uses a random split. | Same recipe on a temporal split: 0.432. Without age and year features: 0.670. | For an unreleased game, expect ordering skill closer to 0.43 than to 0.73. |
| S3 | **Exposure time is mixed with quality.** Owners are a 2019 snapshot; older games had more time to sell. | `game_age_days` is the 3rd most important feature. | Part of what the model "knows" is calendar time, which a developer cannot choose. |
| S4 | **Censored, tied target.** Owners are known only as 13 ranges; 68.7% of games share the lowest one. | Tie-free Spearman ceiling on the test set: 0.835. | Distinctions among small games are mostly invisible to the data. |
| S5 | **Coarse confidence labels.** Confidence compares *medians*, which saturate at the lowest owners range. | Three features get lift 0% and "Low" (chapter 8). | "Low" often means "undetectable with medians", not "harmful". |
| S6 | **No uncertainty per prediction.** The app shows one percentile, without an interval. | Absolute percentile error on the test split: median 13.5, 75th percentile 24.3 points. Two nearly identical training runs (with and without the loader's preprocessing) place the default game at 46.7 and 51.8. | Differences of a few points between two configurations are within the noise. |
| S7 | **One snapshot of Steam.** Data end in May 2019. | Release years 1997 to 2019. | Market changes since then (store algorithms, new genres, other platforms) are not represented. |

## 11.2 Implementation notes (verified on the snapshot run of 2026-10-22)

### Inputs and feature mapping (affect the recommendations)

| # | Behavior | Where | Effect |
|---|---|---|---|
| I1 | The age rating is collected but never passed to the ranker. | `IntelligenceEngine._build_ranker_features` | `required_age` and `is_mature` are always 0. |
| I2 | Genres *Puzzle*, *Horror*; tags *Atmospheric*, *Retro*, *Fantasy*, *Sci-fi*, *Tower Defense*; features *Shared/Split Screen*, *Steam Leaderboards* map to columns the model does not have. | `new_game.py` options vs `_engineer_features` lists | These inputs never change the result. |
| I3 | 39 of 91 model columns cannot be set from the form, including the most important one, `tag_free_to_play`. | `_build_ranker_features` | The form explores only part of what the model learned. |
| I4 | Hidden defaults `release_year = 2024`, `game_age_days = 100`, `dev_game_count = 1`; publisher columns 0. | `_build_ranker_features` | 2024 and a publisher count of 0 lie outside the training range. |
| I5 | *Add Mac & Linux support* sets only `platform_count = 3`. | `get_improvements` | Predicted 53.6 for the default game; applying it through the form gives 55.5. |
| I6 | Binary flags use substring regular-expression matching. | `str.contains(..., case=False)` | `Multi-player` matches `Online Multi-Player`, `RPG` matches `JRPG`. |
| I7 | `game_age_days` is measured from 2017-05-01 although the data reach 2019-05-01. | `_engineer_features` | 55.6% of games have age clipped to 0. |
| I8 | `dev_game_count` and `pub_game_count` count games over the whole file, including later releases. | `_engineer_features` | Mild look-ahead: a studio's future output informs its earlier games. |

### Validation and labels

| # | Behavior | Where | Effect |
|---|---|---|---|
| V1 | `temporal_spearman` is a copy of the random-split value; the note "Based on cross-validation" describes a single holdout. | `_train_ensemble`, `analyze_game` | No temporal evaluation of the ranker exists in the app. |
| V2 | `platform_count` validation compares `== 1` with `== 0`; no game has 0 platforms, so it is never validated. `tag_co_op` has 31 test games (< 50). | `_validate_improvements` | Both appear as "Medium". |
| V3 | `effect_pct` divides a rank difference by the number of test games. | `_validate_improvements` | A tiny number with no direct meaning; not displayed. |
| V4 | "Typical Precision, 75% within" is the 25th percentile of errors. | `models.train_improved_models`, `data_analysis.py` | Only 25% of games are within the displayed distance. |
| V5 | "Ranking Accuracy X%" displays `spearman * 100`. | `data_analysis.py` | A correlation shown as a percentage. |

### Display

| # | Behavior | Where | Effect |
|---|---|---|---|
| D1 | Readiness score can reach at most 70. | `_analyze_success_factors` | "70%" is the best possible readiness. |
| D2 | *Similar Games* is always 0. | `_convert_improvements` | The metric carries no information. |
| D3 | Confidence captions look up lower-case keys. | `new_game.display_analysis_results` | The caption under each improvement is empty. |

### Secondary models and environment

| # | Behavior | Where | Effect |
|---|---|---|---|
| E1 | With **pandas 3.x**, the owners column is not recognized as text (`dtype == 'object'` is false), every game gets 1,000 owners and the ranker's Spearman becomes NaN. | `SteamDataLoader.preprocess_steam_data` | Use pandas 2.x; `requirements.txt` pins `pandas<3`. |
| E2 | Developer-history features are constant or misaligned: `_range_to_midpoint` receives numbers instead of range strings, so `dev_prior_avg_owners` is always 10,000; and rows sorted by date are assigned back by position, so `dev_experience` matches the right game in only 44% of rows. | `ImprovedPreprocessor.compute_developer_history`, `models.enhanced_feature_engineering` | These 5 of the 134 secondary features carry little or wrong information. Ranker unaffected. |
| E3 | Models are not saved to disk; each server start retrains (about 70 s) and each browser session retrains the ranker (about 2 s). | `train_models`, `new_game_page` | Results can differ slightly across XGBoost versions or hardware. |
| E4 | Every training run writes a new `training_report_*.md` into the working directory. | `train_models` | Reports accumulate in the repository folder. |

## 11.3 What would resolve the main uncertainties

| Question | Evidence that would answer it |
|---|---|
| How well does the ranker order *future* games? | Report the temporal-split Spearman (0.432 here) in the app, with a bootstrap interval |
| Are the what-if gains causal? | Natural experiments (for example games that added a feature after launch, compared with similar games that did not), or matching on observed covariates |
| How uncertain is one percentile? | Quantile regression or conformal prediction intervals on the rank score |
| Do confidence labels mean what they say? | Compare mean ranks or the share above 20k owners instead of medians, with a significance test |

---
[Previous: Secondary models and validation](10_secondary_models_and_validation.md) · [Index](README.md) · **11 Limitations**
