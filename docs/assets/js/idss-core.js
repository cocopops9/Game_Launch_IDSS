/*
 * IDSS core: a line-by-line JavaScript port of the decision pipeline.
 *
 * Every function below mirrors a Python function of the repository, so the
 * interactive pages compute the same numbers as the Streamlit app:
 *
 *   buildRankerFeatures  <-  IntelligenceEngine._build_ranker_features  (intelligence_engine.py)
 *   predictRank          <-  IntelligentRanker._predict_rank            (ranking_integration.py)
 *   getPercentile        <-  IntelligentRanker.get_percentile           (ranking_integration.py)
 *   getImprovements      <-  IntelligentRanker.get_improvements         (ranking_integration.py)
 *   analyzeGame          <-  IntelligenceEngine.analyze_game            (intelligence_engine.py)
 *
 * The trees, the reference distribution and the validation results come from
 * docs/assets/data/ranker_model.js, produced by docs/tools/export_lab_data.py.
 */
(function (global) {
  "use strict";

  var MODEL = global.IDSS_MODEL;
  var FEATURES = MODEL.feature_cols;
  var FEATURE_SET = {};
  FEATURES.forEach(function (name) { FEATURE_SET[name] = true; });

  // XGBoost compares features and thresholds in single precision.
  var f32 = Math.fround;

  // Same candidate list, same order, same dependencies as get_improvements().
  var IMPROVEMENT_OPTIONS = [
    ["platform_count", 3, "Add Mac & Linux support", "windows"],
    ["has_achievements", 1, "Add achievements system", null],
    ["achievements", 30, "Add 30+ achievements", "has_achievements"],
    ["cat_steam_trading_cards", 1, "Add Steam Trading Cards", null],
    ["cat_steam_cloud", 1, "Add Steam Cloud saves", null],
    ["cat_multi_player", 1, "Add multiplayer mode", null],
    ["cat_co_op", 1, "Add co-op mode", null],
    ["cat_online_co_op", 1, "Add online co-op", "cat_co_op"],
    ["cat_full_controller_support", 1, "Add full controller support", null],
    ["cat_steam_workshop", 1, "Add Steam Workshop support", null],
    ["tag_multiplayer", 1, "Target multiplayer audience", null],
    ["tag_co_op", 1, "Target co-op audience", null],
    ["tag_story_rich", 1, "Emphasize story elements", null]
  ];

  // Options offered by the Streamlit form in new_game.py.
  var UI_OPTIONS = {
    platforms: ["windows", "mac", "linux"],
    genres: ["Indie", "Action", "Adventure", "Strategy", "Simulation", "RPG", "Casual", "Sports",
             "Racing", "Puzzle", "Horror", "Free to Play", "Early Access", "Massively Multiplayer"],
    tags: ["Singleplayer", "Multiplayer", "Co-op", "Open World", "Survival", "Roguelike", "Sandbox",
           "Story Rich", "Atmospheric", "Horror", "FPS", "Platformer", "Puzzle", "VR",
           "Pixel Graphics", "Retro", "Fantasy", "Sci-fi", "Anime", "Tower Defense"],
    categories: ["Steam Achievements", "Steam Trading Cards", "Steam Cloud", "Steam Workshop",
                 "Full controller support", "Co-op", "Online Co-op", "Shared/Split Screen",
                 "Steam Leaderboards", "VR Support"],
    ageRatings: [0, 13, 16, 18]
  };

  var DEFAULT_GAME = {
    price: 14.99,
    platforms: ["windows"],
    genres: ["Indie"],
    tags: ["Singleplayer"],
    categories: [],
    release_month: 11,
    achievements: 0,
    required_age: 0
  };

  function slug(text) {
    return text.toLowerCase().split(" ").join("_").split("-").join("_");
  }

  function genreColumn(genre) { return "genre_" + genre.toLowerCase().split(" ").join("_"); }
  function tagColumn(tag) { return "tag_" + slug(tag); }
  function categoryColumn(category) { return "cat_" + slug(category); }

  function priceTier(price) {
    if (price === 0) return 0;
    if (price <= 5) return 1;
    if (price <= 10) return 2;
    if (price <= 20) return 3;
    if (price <= 40) return 4;
    return 5;
  }

  /* Convert the form values into the ranker's feature dictionary. */
  function buildRankerFeatures(game) {
    var rf = {};
    var price = game.price || 0;
    rf.price = price;
    rf.price_log = Math.log1p(price);
    rf.is_free = price === 0 ? 1 : 0;
    rf.price_tier = priceTier(price);

    var platforms = game.platforms || ["windows"];
    rf.windows = platforms.indexOf("windows") >= 0 ? 1 : 0;
    rf.mac = platforms.indexOf("mac") >= 0 ? 1 : 0;
    rf.linux = platforms.indexOf("linux") >= 0 ? 1 : 0;
    rf.platform_count = rf.windows + rf.mac + rf.linux;

    var achievements = game.achievements || 0;
    rf.achievements = achievements;
    rf.has_achievements = achievements > 0 ? 1 : 0;
    rf.achievements_log = Math.log1p(achievements);

    // The form does not ask for these, so the Python defaults always apply.
    rf.game_age_days = game.game_age_days !== undefined ? game.game_age_days : 100;
    rf.game_age_log = Math.log1p(rf.game_age_days);
    rf.release_year = game.release_year !== undefined ? game.release_year : 2024;
    rf.release_month = game.release_month !== undefined ? game.release_month : 6;

    (game.genres || []).forEach(function (g) { rf[genreColumn(g)] = 1; });
    (game.tags || []).forEach(function (t) { rf[tagColumn(t)] = 1; });
    (game.categories || []).forEach(function (c) { rf[categoryColumn(c)] = 1; });

    rf.dev_game_count = game.dev_game_count !== undefined ? game.dev_game_count : 1;
    rf.dev_game_count_log = Math.log1p(rf.dev_game_count);
    return rf;
  }

  /* Dense vector in model column order; absent keys become 0, as in _predict_rank(). */
  function toVector(features) {
    return FEATURES.map(function (name) {
      var value = features[name];
      return (value === undefined || value === null || !isFinite(value)) ? 0 : value;
    });
  }

  function evaluateTree(tree, vector) {
    var node = tree.root;
    while (tree.f[node] !== -1) {
      node = f32(vector[tree.f[node]]) < f32(tree.t[node]) ? tree.y[node] : tree.n[node];
    }
    return { leaf: node, value: tree.t[node] };
  }

  /* Raw rank score = base_score + sum of the 150 leaf values. */
  function predictRankFromVector(vector, keepTrace) {
    var total = MODEL.base_score;
    var trace = keepTrace ? [] : null;
    for (var i = 0; i < MODEL.trees.length; i++) {
      var result = evaluateTree(MODEL.trees[i], vector);
      total += result.value;
      if (trace) trace.push(result.value);
    }
    return keepTrace ? { score: total, contributions: trace } : total;
  }

  function predictRank(features) {
    return predictRankFromVector(toVector(features), false);
  }

  /* np.searchsorted(sorted_reference, score) / n * 100, clipped to [1, 99]. */
  function countBelow(sorted, value) {
    var lo = 0, hi = sorted.length;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (sorted[mid] < value) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  function scoreToPercentile(score) {
    var ref = MODEL.reference_ranks;
    var raw = countBelow(ref, score) / ref.length * 100;
    return Math.min(99, Math.max(1, raw));
  }

  function getPercentile(features) {
    return scoreToPercentile(predictRank(features));
  }

  function round1(x) { return Math.round(x * 10) / 10; }

  function getImprovements(features, topN) {
    var base = getPercentile(features);
    var results = [];
    var evaluated = [];

    IMPROVEMENT_OPTIONS.forEach(function (option) {
      var feature = option[0], target = option[1], description = option[2], dependency = option[3];
      var record = { feature: feature, description: description, status: "", gain: null };
      evaluated.push(record);

      if (!FEATURE_SET[feature]) { record.status = "not a model feature"; return; }
      var current = features[feature] || 0;
      if (current >= target) { record.status = "already present"; return; }
      if (dependency && (features[dependency] || 0) < 1) { record.status = "needs " + dependency; return; }

      var modified = Object.assign({}, features);
      modified[feature] = target;
      var newPercentile = getPercentile(modified);
      var gain = newPercentile - base;
      record.gain = gain;

      var validation = MODEL.validated_improvements[feature] || {};
      var confidence = validation.confidence || "Medium";
      var lift = validation.actual_lift || 0;

      if (gain > 0.5) {
        record.status = "kept";
        results.push({
          feature: feature,
          description: description,
          percentile_gain: round1(gain),
          new_percentile: round1(newPercentile),
          confidence: confidence,
          historical_lift: lift ? (lift > 0 ? "+" : "") + Math.round(lift) + "%" : "N/A",
          current_value: current,
          suggested_value: target
        });
      } else {
        record.status = "gain <= 0.5 pp (discarded)";
      }
    });

    results.sort(function (a, b) { return b.percentile_gain - a.percentile_gain; });
    return { kept: results.slice(0, topN || 5), evaluated: evaluated, base: base };
  }

  function tierFor(percentile) {
    if (percentile >= 95) return "exceptional";
    if (percentile >= 80) return "strong";
    if (percentile >= 60) return "above_average";
    if (percentile >= 40) return "average";
    if (percentile >= 20) return "below_average";
    return "struggling";
  }

  function assessRisks(game, percentile) {
    var risks = [];
    if ((game.platforms || []).length === 1) {
      risks.push({ level: "moderate", title: "Limited Platform Support",
                   description: "Single platform limits audience.", recommendation: "Consider Mac/Linux ports." });
    }
    if ((game.price || 0) > 40) {
      risks.push({ level: "high", title: "Premium Pricing Risk",
                   description: "Competing with AAA titles.", recommendation: "Ensure clear value proposition." });
    }
    if (percentile < 30) {
      risks.push({ level: "high", title: "Below-Average Position",
                   description: "In bottom third of market.", recommendation: "Review improvement scenarios." });
    }
    return risks;
  }

  function successFactors(game) {
    var present = [], missing = [];
    ((game.platforms || []).length >= 2 ? present : missing).push({ factor: "Multi-platform", impact: "high" });
    ((game.achievements || 0) > 0 ? present : missing).push({ factor: "Achievements", impact: "medium" });
    var cats = (game.categories || []).map(function (c) { return c.toLowerCase(); });
    (cats.indexOf("steam trading cards") >= 0 ? present : missing).push({ factor: "Trading Cards", impact: "medium" });
    var score = present.reduce(function (sum, f) { return sum + (f.impact === "high" ? 3 : 2); }, 0);
    return { present: present, missing: missing, readiness_score: Math.min(100, score * 10) };
  }

  function insightsFor(percentile, improvements) {
    var insights = [];
    if (percentile >= 70) {
      insights.push({ type: "success", title: "Strong Position",
        message: "Top " + (100 - percentile).toFixed(0) + "% of market. High confidence ranking (Spearman: " + MODEL.spearman.toFixed(2) + ")" });
    } else if (percentile >= 40) {
      insights.push({ type: "info", title: "Average Position", message: percentile.toFixed(0) + "th percentile. See improvements below." });
    } else {
      insights.push({ type: "warning", title: "Below Average", message: "Bottom " + percentile.toFixed(0) + "%. Review recommendations." });
    }
    if (improvements.length) {
      var top = improvements[0];
      insights.push({ type: "opportunity", title: top.description,
        message: "+" + top.percentile_gain.toFixed(1) + "pp → " + top.new_percentile.toFixed(0) + "th. Confidence: " + top.confidence });
    }
    return insights;
  }

  function analyzeGame(game) {
    var features = buildRankerFeatures(game);
    var trace = predictRankFromVector(toVector(features), true);
    var percentile = scoreToPercentile(trace.score);
    var improvements = getImprovements(features, 6);
    return {
      features: features,
      score: trace.score,
      contributions: trace.contributions,
      percentile: percentile,
      tier: tierFor(percentile),
      improvements: improvements.kept,
      evaluated: improvements.evaluated,
      risks: assessRisks(game, percentile),
      success_factors: successFactors(game),
      insights: insightsFor(percentile, improvements.kept)
    };
  }

  /* Which form inputs reach the model, and which are silently dropped. */
  function inputCoverage(game) {
    var rows = [];
    (game.genres || []).forEach(function (g) { rows.push({ input: "Genre: " + g, column: genreColumn(g), used: !!FEATURE_SET[genreColumn(g)] }); });
    (game.tags || []).forEach(function (t) { rows.push({ input: "Tag: " + t, column: tagColumn(t), used: !!FEATURE_SET[tagColumn(t)] }); });
    (game.categories || []).forEach(function (c) { rows.push({ input: "Feature: " + c, column: categoryColumn(c), used: !!FEATURE_SET[categoryColumn(c)] }); });
    rows.push({ input: "Age rating: " + (game.required_age || 0), column: "required_age", used: false,
                note: "the column exists, but _build_ranker_features never copies it, so the model always sees 0" });
    return rows;
  }

  global.IDSS = {
    MODEL: MODEL,
    FEATURES: FEATURES,
    FEATURE_SET: FEATURE_SET,
    UI_OPTIONS: UI_OPTIONS,
    DEFAULT_GAME: DEFAULT_GAME,
    IMPROVEMENT_OPTIONS: IMPROVEMENT_OPTIONS,
    buildRankerFeatures: buildRankerFeatures,
    toVector: toVector,
    evaluateTree: evaluateTree,
    predictRank: predictRank,
    predictRankFromVector: predictRankFromVector,
    scoreToPercentile: scoreToPercentile,
    getPercentile: getPercentile,
    getImprovements: getImprovements,
    tierFor: tierFor,
    analyzeGame: analyzeGame,
    inputCoverage: inputCoverage,
    columnNames: { genre: genreColumn, tag: tagColumn, category: categoryColumn }
  };
})(typeof window !== "undefined" ? window : globalThis);
