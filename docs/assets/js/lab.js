/*
 * Interactive Lab: UI layer.
 *
 * All model numbers come from IDSS (idss-core.js), which runs the exported
 * XGBoost trees. This file only renders: forms, charts and animations.
 */
(function () {
  "use strict";

  var M = window.IDSS_MODEL;
  var D = window.IDSS_DATA;
  var I = window.IDSS;
  var SVGNS = "http://www.w3.org/2000/svg";

  // --------------------------------------------------------------------------------------
  // Small helpers
  // --------------------------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    setAttrs(node, attrs);
    appendChildren(node, children);
    return node;
  }

  function svgEl(tag, attrs, children) {
    var node = document.createElementNS(SVGNS, tag);
    setAttrs(node, attrs);
    appendChildren(node, children);
    return node;
  }

  function setAttrs(node, attrs) {
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === "text") node.textContent = attrs[key];
      else if (key === "html") node.innerHTML = attrs[key];
      else if (key === "class") node.setAttribute("class", attrs[key]);
      else if (key.indexOf("on") === 0 && typeof attrs[key] === "function") node.addEventListener(key.slice(2), attrs[key]);
      else if (attrs[key] !== null && attrs[key] !== undefined) node.setAttribute(key, attrs[key]);
    });
  }

  function appendChildren(node, children) {
    (children || []).forEach(function (child) {
      if (child === null || child === undefined) return;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    });
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }
  function fmtInt(n) { return Math.round(n).toLocaleString("en-US"); }
  function pct(x, digits) { return (x * 100).toFixed(digits === undefined ? 1 : digits) + "%"; }
  function ordinal(p) { return p.toFixed(1) + "th"; }

  function scale(d0, d1, r0, r1) {
    return function (v) { return r0 + (r1 - r0) * (v - d0) / (d1 - d0); };
  }

  function chart(width, height) {
    return svgEl("svg", { viewBox: "0 0 " + width + " " + height, class: "chart", preserveAspectRatio: "xMidYMid meet" });
  }

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  // Deterministic pseudo-random numbers, so sliders give stable pictures.
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function gaussian(rand) {
    var u = Math.max(rand(), 1e-12), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // --------------------------------------------------------------------------------------
  // Global state: one game configuration drives every section
  // --------------------------------------------------------------------------------------
  var state = {
    game: JSON.parse(JSON.stringify(I.DEFAULT_GAME)),
    analysis: null,
    previousVector: null,
    pendingApply: null,
    applyNote: null
  };

  var PRESETS = [
    { name: "Streamlit defaults", game: I.DEFAULT_GAME },
    { name: "Feature-rich co-op indie", game: { price: 19.99, platforms: ["windows", "mac", "linux"], genres: ["Action", "Indie"],
        tags: ["Multiplayer", "Co-op"], categories: ["Steam Achievements", "Steam Trading Cards", "Steam Cloud", "Full controller support", "Co-op", "Online Co-op"],
        release_month: 11, achievements: 40, required_age: 0 } },
    { name: "Free-to-play MMO", game: { price: 0, platforms: ["windows"], genres: ["Free to Play", "Action", "Massively Multiplayer"],
        tags: ["Multiplayer"], categories: [], release_month: 6, achievements: 0, required_age: 0 } },
    { name: "Premium single-player RPG", game: { price: 49.99, platforms: ["windows"], genres: ["Action", "Adventure", "RPG"],
        tags: ["Singleplayer", "Open World", "Story Rich"], categories: ["Steam Achievements", "Steam Cloud", "Full controller support"],
        release_month: 10, achievements: 50, required_age: 18 } }
  ];

  function update() {
    state.analysis = I.analyzeGame(state.game);
    renderForm();
    renderFeatures();
    renderRealTrees();
    renderCdf();
    renderWhatIf();
    renderDashboard();
    var a = state.analysis;
    $("badge-pct").textContent = ordinal(a.percentile);
    $("badge-tier").textContent = a.tier.replace("_", " ");
  }

  // --------------------------------------------------------------------------------------
  // Step 1: pipeline stages
  // --------------------------------------------------------------------------------------
  var STAGES = [
    { t: "Load data", s: "steam.csv, 27,075 rows", target: "data",
      d: "<b>SteamDataLoader.load_steam_data</b> reads <code>steam.csv</code>, converts each SteamSpy owners range to its midpoint and computes the review ratio. " +
         "The IntelligentRanker then trains on this dataframe." },
    { t: "Engineer features", s: "91 numeric columns", target: "features",
      d: "<b>IntelligentRanker._engineer_features</b> creates 91 columns in 10 families: price (4), time (4), platforms (4), achievements (3), age rating (2), " +
         "categories (12), genres (12), SteamSpy tags (26), developer (22) and publisher (2). Only information known before launch is used." },
    { t: "Train the ranker", s: "XGBoost, 150 trees", target: "boosting",
      d: "A random 80/20 split (<code>random_state=42</code>) gives 21,660 training and 5,415 test games. The target is <code>rankdata(owners) / n</code>. " +
         "<code>XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, min_child_weight=3, subsample=0.8, colsample_bytree=0.8)</code>." },
    { t: "Score your game", s: "sum of 150 leaves", target: "boosting",
      d: "<b>_build_ranker_features</b> maps the form to columns, <b>_predict_rank</b> evaluates the 150 trees. The raw score lives on the scale of the rank target (about 0.24 to 1.13 on the test split)." },
    { t: "Percentile", s: "vs 5,415 held-out games", target: "percentile",
      d: "<b>get_percentile</b> counts the reference scores (model predictions on the test split) below your score, divides by 5,415 and clips the result to [1, 99]." },
    { t: "Decide", s: "what-ifs, tier, risks", target: "whatif",
      d: "<b>get_improvements</b> re-scores 13 one-feature changes. <b>IntelligenceEngine</b> adds the tier, risk rules, success factors, readiness score and insights shown by the app." }
  ];

  function renderStages() {
    var host = $("stages");
    var detail = $("stage-detail");
    var current = 0;
    var auto = true;

    function show(i) {
      current = i;
      Array.prototype.forEach.call(host.children, function (node, k) { node.classList.toggle("on", k === i); });
      var s = STAGES[i];
      detail.innerHTML = "<div class='kicker'>Stage " + (i + 1) + " of 6</div><p style='margin:6px 0 10px'>" + s.d + "</p>";
      detail.appendChild(el("a", { href: "#" + s.target, text: "Go to the step that shows it" }));
    }

    STAGES.forEach(function (s, i) {
      host.appendChild(el("div", { class: "stage", onclick: function () { auto = false; show(i); } }, [
        el("div", { class: "n", text: "0" + (i + 1) }),
        el("div", { class: "t", text: s.t }),
        el("div", { class: "s", text: s.s })
      ]));
    });
    show(0);
    setInterval(function () { if (auto) show((current + 1) % STAGES.length); }, 4200);

    var maxDiff = 0;
    M.check_rows.forEach(function (row, i) {
      maxDiff = Math.max(maxDiff, Math.abs(I.predictRankFromVector(row) - M.check_predictions[i]));
    });
    $("model-facts").innerHTML =
      "<b>Model snapshot</b> " + M.generated + " (xgboost " + M.xgboost_version + ", pandas " + M.pandas_version + "): " +
      fmtInt(M.n_train) + " training games, " + fmtInt(M.n_test) + " test games, Spearman " + M.spearman.toFixed(3) + ". " +
      "Self-check just run in your browser: on 25 held-out rows the JavaScript evaluator matches Python's <code>model.predict</code> " +
      "with a maximum absolute difference of <b>" + maxDiff.toExponential(1) + "</b>.";
    $("foot-date").textContent = M.generated;
  }

  // --------------------------------------------------------------------------------------
  // Step 2: data
  // --------------------------------------------------------------------------------------
  function renderDataStats() {
    var lowest = D.owners_tiers[0].count / D.n_games;
    var stats = [
      [fmtInt(D.n_games), "games in steam.csv"],
      [pct(lowest), "in the lowest range (0 to 20k owners)"],
      [pct(D.share_free), "free to play (price 0)"],
      [pct(D.share_age_clipped_to_zero), "released after 2017-05-01, so game_age_days is clipped to 0"]
    ];
    var host = $("data-stats");
    stats.forEach(function (s) {
      host.appendChild(el("div", { class: "stat" }, [el("div", { class: "v", text: s[0] }), el("div", { class: "l", text: s[1] })]));
    });
  }

  function renderOwnersChart(mode) {
    var host = clear($("chart-owners"));
    var W = 520, H = 330, left = 92, right = 60, top = 8, rowH = 24;
    var svg = chart(W, H);
    var maxV = D.owners_tiers[0].count;
    var x = mode === "log"
      ? function (v) { return left + (W - left - right) * Math.log10(v + 1) / Math.log10(maxV + 1); }
      : function (v) { return left + (W - left - right) * v / maxV; };
    D.owners_tiers.forEach(function (t, i) {
      var y = top + i * rowH;
      svg.appendChild(svgEl("text", { x: left - 8, y: y + 15, "text-anchor": "end", class: "mono", text: t.label }));
      var bar = svgEl("rect", { x: left, y: y + 4, height: 15, rx: 3, width: 0, fill: i === 0 ? css("--cyan") : css("--violet") });
      svg.appendChild(bar);
      svg.appendChild(svgEl("text", { x: x(t.count) + 6, y: y + 15, text: fmtInt(t.count) }));
      requestAnimationFrame(function () {
        bar.style.transition = "width 0.8s cubic-bezier(.2,.8,.2,1) " + (i * 40) + "ms";
        bar.setAttribute("width", Math.max(2, x(t.count) - left));
      });
    });
    host.appendChild(svg);
  }

  function renderYearsChart(mode) {
    var host = clear($("chart-years"));
    var years = Object.keys(D.release_years).map(Number).filter(function (y) { return y >= 2004; });
    var W = 520, H = 300, left = 46, bottom = 40, top = 10;
    var maxV = Math.max.apply(null, years.map(function (y) { return D.release_years[y]; }));
    var bw = (W - left - 10) / years.length;
    var y = scale(0, maxV, H - bottom, top);
    var svg = chart(W, H);
    [0, 2000, 4000, 6000, 8000].forEach(function (v) {
      svg.appendChild(svgEl("line", { x1: left, x2: W - 10, y1: y(v), y2: y(v), class: "grid" }));
      svg.appendChild(svgEl("text", { x: left - 6, y: y(v) + 4, "text-anchor": "end", text: fmtInt(v) }));
    });
    var green = css("--cyan"), orange = css("--orange");
    years.forEach(function (yr, i) {
      var n = D.release_years[yr];
      var x0 = left + i * bw + 2;
      var trainShare = mode === "random" ? 0.8 : (yr < 2018 ? 1 : 0);
      var hTrain = (H - bottom - y(n)) * trainShare;
      var hTest = (H - bottom - y(n)) * (1 - trainShare);
      svg.appendChild(svgEl("rect", { x: x0, width: bw - 4, y: H - bottom - hTrain, height: hTrain, fill: green, rx: 2 }));
      svg.appendChild(svgEl("rect", { x: x0, width: bw - 4, y: H - bottom - hTrain - hTest, height: hTest, fill: orange, rx: 2 }));
      if (yr % 2 === 0) svg.appendChild(svgEl("text", { x: x0 + bw / 2, y: H - bottom + 16, "text-anchor": "middle", text: String(yr) }));
    });
    if (mode === "temporal") {
      var cut = left + years.indexOf(2018) * bw;
      svg.appendChild(svgEl("line", { x1: cut, x2: cut, y1: top, y2: H - bottom, stroke: css("--text"), "stroke-dasharray": "4 4" }));
      svg.appendChild(svgEl("text", { x: cut - 6, y: top + 12, "text-anchor": "end", class: "t-strong", text: "cutoff 2018-01-01" }));
    }
    host.appendChild(svg);
    host.appendChild(el("div", { class: "legend" }, [
      el("span", {}, [el("i", { style: "background:" + green }), "training"]),
      el("span", {}, [el("i", { style: "background:" + orange }), "test"])
    ]));
    $("split-note").innerHTML = mode === "random"
      ? "IntelligentRanker: <code>train_test_split(test_size=0.2, random_state=42)</code>, 21,660 / 5,415 games. Every year appears in both sets (bars show the expected 80/20 share)."
      : "models.train_improved_models: games released before 2018 train (16,702), games from 2018 test (10,373). This simulates predicting future launches.";
  }

  function bindSeg(id, onChange) {
    var seg = $(id);
    seg.addEventListener("click", function (e) {
      if (e.target.tagName !== "BUTTON") return;
      Array.prototype.forEach.call(seg.children, function (b) { b.classList.toggle("on", b === e.target); });
      onChange(e.target.getAttribute("data-v"));
    });
  }

  // --------------------------------------------------------------------------------------
  // Step 3: the form
  // --------------------------------------------------------------------------------------
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function buildForm() {
    var presets = $("presets");
    presets.appendChild(el("span", { class: "muted", text: "presets:" }));
    PRESETS.forEach(function (p) {
      presets.appendChild(el("button", { class: "btn small", type: "button", text: p.name, onclick: function () {
        state.game = JSON.parse(JSON.stringify(p.game));
        state.applyNote = null;
        update();
      } }));
    });

    $("in-price").addEventListener("input", function (e) { state.game.price = parseFloat(e.target.value); update(); });
    $("in-achievements").addEventListener("input", function (e) {
      var before = state.game.achievements;
      var value = Math.max(0, Math.min(500, parseInt(e.target.value || "0", 10)));
      state.game.achievements = value;
      // The Streamlit checkbox defaults to "achievements > 0".
      if ((before > 0) !== (value > 0)) toggleInList(state.game.categories, "Steam Achievements", value > 0);
      update();
    });
    var month = $("in-month");
    MONTHS.forEach(function (m, i) { month.appendChild(el("option", { value: i + 1, text: m })); });
    month.addEventListener("change", function (e) { state.game.release_month = parseInt(e.target.value, 10); update(); });
    var age = $("in-age");
    I.UI_OPTIONS.ageRatings.forEach(function (a) { age.appendChild(el("option", { value: a, text: a === 0 ? "Everyone" : a + "+" })); });
    age.addEventListener("change", function (e) { state.game.required_age = parseInt(e.target.value, 10); update(); });
  }

  function toggleInList(list, value, on) {
    var i = list.indexOf(value);
    if (on && i < 0) list.push(value);
    if (!on && i >= 0) list.splice(i, 1);
  }

  function chips(hostId, options, list, columnOf) {
    var host = clear($(hostId));
    options.forEach(function (opt) {
      var ignored = columnOf && !I.FEATURE_SET[columnOf(opt)];
      var chip = el("span", {
        class: "chip" + (list.indexOf(opt) >= 0 ? " on" : "") + (ignored ? " ignored" : ""),
        title: columnOf ? (ignored ? columnOf(opt) + " is not one of the 91 model columns" : "sets " + columnOf(opt) + " = 1") : opt,
        text: opt,
        onclick: function () { toggleInList(list, opt, list.indexOf(opt) < 0); update(); }
      });
      host.appendChild(chip);
    });
  }

  function renderForm() {
    var g = state.game;
    $("in-price").value = g.price;
    $("price-out").textContent = "$" + g.price.toFixed(2);
    $("in-achievements").value = g.achievements;
    $("in-month").value = g.release_month;
    $("in-age").value = g.required_age;
    chips("in-platforms", I.UI_OPTIONS.platforms, g.platforms, null);
    chips("in-genres", I.UI_OPTIONS.genres, g.genres, I.columnNames.genre);
    chips("in-tags", I.UI_OPTIONS.tags, g.tags, I.columnNames.tag);

    var host = clear($("in-categories"));
    I.UI_OPTIONS.categories.forEach(function (c) {
      var ignored = !I.FEATURE_SET[I.columnNames.category(c)];
      var box = el("input", { type: "checkbox" });
      box.checked = g.categories.indexOf(c) >= 0;
      box.addEventListener("change", function () { toggleInList(g.categories, c, box.checked); update(); });
      host.appendChild(el("label", { class: ignored ? "ignored" : "", title: I.columnNames.category(c) }, [box, el("span", { text: c })]));
    });
  }

  // --------------------------------------------------------------------------------------
  // Step 4: feature grid and coverage
  // --------------------------------------------------------------------------------------
  var FAMILIES = [
    ["Price", "--cyan", function (f) { return ["price", "price_log", "is_free", "price_tier"].indexOf(f) >= 0; }],
    ["Time", "--violet", function (f) { return ["game_age_days", "game_age_log", "release_year", "release_month"].indexOf(f) >= 0; }],
    ["Platforms", "--green", function (f) { return ["windows", "mac", "linux", "platform_count"].indexOf(f) >= 0; }],
    ["Achievements", "--amber", function (f) { return f.indexOf("achievements") >= 0; }],
    ["Age rating", "--pink", function (f) { return f === "required_age" || f === "is_mature"; }],
    ["Categories", "--cyan", function (f) { return f.indexOf("cat_") === 0; }],
    ["Genres", "--violet", function (f) { return f.indexOf("genre_") === 0; }],
    ["Tags", "--green", function (f) { return f.indexOf("tag_") === 0; }],
    ["Developer", "--amber", function (f) { return f.indexOf("dev_") === 0; }],
    ["Publisher", "--pink", function (f) { return f.indexOf("pub_") === 0; }]
  ];
  var BINARY = /^(is_free|windows|mac|linux|has_achievements|is_mature|cat_|genre_|tag_|dev_top_)/;

  function renderFeatures() {
    var vector = I.toVector(state.analysis.features);
    var prev = state.previousVector;
    var host = clear($("feature-grid"));
    FAMILIES.forEach(function (fam) {
      var cols = I.FEATURES.filter(fam[2]);
      var cells = el("div", { class: "cells", style: "--fam: var(" + fam[1] + ")" });
      cols.forEach(function (name) {
        var idx = I.FEATURES.indexOf(name);
        var v = vector[idx];
        var isBinary = BINARY.test(name);
        var label = isBinary ? "" : (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(2));
        var cell = el("span", { class: "cell" + (v !== 0 ? " on" : ""), title: name + " = " + (Math.abs(v - Math.round(v)) < 1e-9 ? v : v.toFixed(4)), text: label });
        if (prev && prev[idx] !== v) {
          cell.classList.add("flash");
          setTimeout(function () { cell.classList.remove("flash"); }, 350);
        }
        cells.appendChild(cell);
      });
      host.appendChild(el("div", { class: "fam" }, [
        el("div", { class: "fam-title", style: "color: var(" + fam[1] + ")", text: fam[0] + " : " + cols.length }), cells
      ]));
    });
    var nonzero = vector.filter(function (v) { return v !== 0; }).length;
    host.appendChild(el("p", { class: "source", text: nonzero + " of 91 columns are non-zero for this configuration." }));
    state.previousVector = vector;

    var table = clear($("coverage"));
    table.appendChild(el("tr", {}, [el("th", { text: "form input" }), el("th", { text: "model column" }), el("th", { text: "reaches model?" })]));
    var g = state.game;
    var fixed = [
      { input: "Price $" + g.price.toFixed(2), column: "price, price_log, is_free, price_tier", used: true },
      { input: "Platforms: " + g.platforms.join(", "), column: "windows, mac, linux, platform_count", used: true },
      { input: "Achievements: " + g.achievements, column: "achievements, has_achievements, achievements_log", used: true },
      { input: "Release month: " + MONTHS[g.release_month - 1], column: "release_month", used: true }
    ];
    fixed.concat(I.inputCoverage(g)).forEach(function (row) {
      table.appendChild(el("tr", {}, [
        el("td", { text: row.input }),
        el("td", { class: "mono", text: row.column }),
        el("td", { html: row.used ? "<span class='tag-ok'>yes</span>" : "<span class='tag-no'>no</span>" + (row.note ? "<div class='source'>" + row.note + "</div>" : "<div class='source'>not one of the 91 columns</div>") })
      ]));
    });
  }

  // --------------------------------------------------------------------------------------
  // Step 5A: toy gradient boosting, fitted live
  // --------------------------------------------------------------------------------------
  var TOY = {
    x: [0.015, 0.856, 0.941, 1.137, 1.597, 2.368, 2.842, 2.927, 2.984, 3.14, 3.912, 4.306, 4.331, 4.713, 4.791, 5.167, 5.822, 5.852, 5.868, 6.485, 6.962, 7.346, 7.378, 7.733, 8.013, 8.917, 9.563, 9.735],
    y: [-0.064, 1.492, 0.925, 1.331, 2.08, 2.075, 1.679, 1.821, 0.553, 1.771, 0.27, -0.416, 0.238, 0.011, -0.457, -0.941, -0.715, -0.74, -0.228, -0.188, 0.065, 0.858, 0.441, 0.681, 1.609, 2.744, 2.926, 2.783]
  };

  // Least-squares regression tree on one feature (depth-limited), as used inside boosting.
  function fitTree(xs, rs, idx, depth) {
    var mean = idx.reduce(function (s, i) { return s + rs[i]; }, 0) / idx.length;
    if (depth === 0 || idx.length < 2) return { leaf: mean };
    var sorted = idx.slice().sort(function (a, b) { return xs[a] - xs[b]; });
    var total = 0, totalSq = 0;
    sorted.forEach(function (i) { total += rs[i]; totalSq += rs[i] * rs[i]; });
    var best = null, leftSum = 0, leftSq = 0;
    for (var k = 0; k < sorted.length - 1; k++) {
      leftSum += rs[sorted[k]]; leftSq += rs[sorted[k]] * rs[sorted[k]];
      if (xs[sorted[k]] === xs[sorted[k + 1]]) continue;
      var nl = k + 1, nr = sorted.length - nl;
      var sse = (leftSq - leftSum * leftSum / nl) + ((totalSq - leftSq) - (total - leftSum) * (total - leftSum) / nr);
      if (!best || sse < best.sse) best = { sse: sse, k: k, threshold: (xs[sorted[k]] + xs[sorted[k + 1]]) / 2 };
    }
    if (!best) return { leaf: mean };
    return {
      threshold: best.threshold,
      left: fitTree(xs, rs, sorted.slice(0, best.k + 1), depth - 1),
      right: fitTree(xs, rs, sorted.slice(best.k + 1), depth - 1)
    };
  }

  function predictTree(tree, x) {
    while (tree.leaf === undefined) tree = x < tree.threshold ? tree.left : tree.right;
    return tree.leaf;
  }

  var toy = { lr: 0.08, m: 0, grid: [], frames: [], losses: [] };

  function trainToy() {
    var n = TOY.x.length, idx = [];
    for (var i = 0; i < n; i++) idx.push(i);
    toy.grid = [];
    for (var g = 0; g <= 120; g++) toy.grid.push(g / 12);
    var mean = TOY.y.reduce(function (s, v) { return s + v; }, 0) / n;
    var F = TOY.y.map(function () { return mean; });
    var Fgrid = toy.grid.map(function () { return mean; });
    toy.frames = [{ F: F.slice(), grid: Fgrid.slice() }];
    toy.losses = [mse(F)];
    for (var m = 1; m <= 150; m++) {
      var residuals = TOY.y.map(function (y, k) { return y - F[k]; });
      var tree = fitTree(TOY.x, residuals, idx, 2);
      F = F.map(function (f, k) { return f + toy.lr * predictTree(tree, TOY.x[k]); });
      Fgrid = Fgrid.map(function (f, k) { return f + toy.lr * predictTree(tree, toy.grid[k]); });
      toy.frames.push({ F: F.slice(), grid: Fgrid.slice() });
      toy.losses.push(mse(F));
    }
  }

  function mse(F) {
    return TOY.y.reduce(function (s, y, k) { return s + (y - F[k]) * (y - F[k]); }, 0) / TOY.y.length;
  }

  function renderToy() {
    var frame = toy.frames[toy.m];
    $("toy-m-out").textContent = toy.m;
    var W = 460, H = 280, p = 30;
    var x = scale(0, 10, p, W - 10), y = scale(-1.4, 3.3, H - p, 10);
    var svg = chart(W, H);
    svg.appendChild(svgEl("rect", { x: p, y: 10, width: W - p - 10, height: H - p - 10, fill: css("--panel-2"), rx: 8 }));
    TOY.x.forEach(function (xv, k) {
      svg.appendChild(svgEl("line", { x1: x(xv), x2: x(xv), y1: y(TOY.y[k]), y2: y(frame.F[k]), stroke: css("--red"), "stroke-width": 1.5, opacity: 0.7 }));
    });
    var d = toy.grid.map(function (g, k) { return (k ? "L" : "M") + x(g).toFixed(1) + "," + y(frame.grid[k]).toFixed(1); }).join(" ");
    svg.appendChild(svgEl("path", { d: d, fill: "none", stroke: css("--amber"), "stroke-width": 3 }));
    TOY.x.forEach(function (xv, k) { svg.appendChild(svgEl("circle", { cx: x(xv), cy: y(TOY.y[k]), r: 4.5, fill: css("--cyan") })); });
    svg.appendChild(svgEl("text", { x: p + 10, y: 30, class: "t-strong", text: "F_M(x) after " + toy.m + " trees (red: residuals)" }));
    clear($("toy-fit")).appendChild(svg);

    var L = chart(W, H);
    var lx = scale(0, 150, 44, W - 10), ly = scale(0, toy.losses[0], H - p, 20);
    L.appendChild(svgEl("line", { x1: 44, x2: W - 10, y1: H - p, y2: H - p, class: "axis" }));
    L.appendChild(svgEl("line", { x1: 44, x2: 44, y1: 20, y2: H - p, class: "axis" }));
    var ld = toy.losses.map(function (v, m) { return (m ? "L" : "M") + lx(m).toFixed(1) + "," + ly(v).toFixed(1); }).join(" ");
    L.appendChild(svgEl("path", { d: ld, fill: "none", stroke: css("--green"), "stroke-width": 2.5 }));
    L.appendChild(svgEl("circle", { cx: lx(toy.m), cy: ly(toy.losses[toy.m]), r: 6, fill: css("--amber") }));
    [0, 50, 100, 150].forEach(function (m) { L.appendChild(svgEl("text", { x: lx(m), y: H - p + 16, "text-anchor": "middle", text: String(m) })); });
    L.appendChild(svgEl("text", { x: 50, y: 14, class: "t-strong", text: "training MSE = " + toy.losses[toy.m].toFixed(3) }));
    clear($("toy-loss")).appendChild(L);
  }

  function bindToy() {
    trainToy();
    renderToy();
    $("toy-m").addEventListener("input", function (e) { toy.m = parseInt(e.target.value, 10); renderToy(); });
    bindSeg("toy-lr", function (v) { toy.lr = parseFloat(v); trainToy(); renderToy(); });
    var timer = null;
    $("toy-play").addEventListener("click", function () {
      if (timer) { clearInterval(timer); timer = null; $("toy-play").textContent = "Play"; return; }
      if (toy.m >= 150) toy.m = 0;
      $("toy-play").textContent = "Pause";
      timer = setInterval(function () {
        toy.m = Math.min(150, toy.m + 1);
        $("toy-m").value = toy.m;
        renderToy();
        if (toy.m >= 150) { clearInterval(timer); timer = null; $("toy-play").textContent = "Play"; }
      }, 45);
    });
  }

  // --------------------------------------------------------------------------------------
  // Step 5B: the real 150 trees for the current configuration
  // --------------------------------------------------------------------------------------
  function renderRealTrees() {
    var a = state.analysis;
    var W = 520, H = 260, left = 50, bottom = 26;
    var cum = [M.base_score];
    a.contributions.forEach(function (c) { cum.push(cum[cum.length - 1] + c); });
    var lo = Math.min.apply(null, cum.concat([M.base_score])) - 0.02;
    var hi = Math.max.apply(null, cum.concat([M.base_score])) + 0.02;
    var x = scale(0, 150, left, W - 10), y = scale(lo, hi, H - bottom, 10);
    var maxAbs = Math.max.apply(null, a.contributions.map(Math.abs));
    var svg = chart(W, H);
    svg.appendChild(svgEl("line", { x1: left, x2: W - 10, y1: y(M.base_score), y2: y(M.base_score), class: "grid" }));
    svg.appendChild(svgEl("text", { x: left - 6, y: y(M.base_score) + 4, "text-anchor": "end", text: "base " + M.base_score.toFixed(2) }));
    var barBase = H - bottom - 30;
    a.contributions.forEach(function (c, i) {
      var h = 28 * Math.abs(c) / maxAbs;
      svg.appendChild(svgEl("rect", { x: x(i + 0.5) - 1, width: 2, y: c >= 0 ? barBase - h : barBase, height: h, fill: c >= 0 ? css("--green") : css("--orange"), opacity: 0.8 }));
    });
    var d = cum.map(function (v, i) { return (i ? "L" : "M") + x(i).toFixed(1) + "," + y(v).toFixed(1); }).join(" ");
    var path = svgEl("path", { d: d, fill: "none", stroke: css("--amber"), "stroke-width": 2.5 });
    svg.appendChild(path);
    svg.appendChild(svgEl("circle", { cx: x(150), cy: y(a.score), r: 5, fill: css("--amber") }));
    svg.appendChild(svgEl("text", { x: x(150) - 6, y: y(a.score) - 10, "text-anchor": "end", class: "t-strong", text: "final score " + a.score.toFixed(4) }));
    [0, 50, 100, 150].forEach(function (m) { svg.appendChild(svgEl("text", { x: x(m), y: H - 8, "text-anchor": "middle", text: String(m) })); });
    clear($("tree-cumulative")).appendChild(svg);
    var length = path.getTotalLength ? path.getTotalLength() : 1000;
    path.style.strokeDasharray = length;
    path.style.strokeDashoffset = length;
    requestAnimationFrame(function () {
      path.style.transition = "stroke-dashoffset 1.1s ease-out";
      path.style.strokeDashoffset = 0;
    });
    renderTreePath();
  }

  function renderTreePath() {
    var t = parseInt($("tree-idx").value, 10) - 1;
    $("tree-idx-out").textContent = "#" + (t + 1);
    var tree = M.trees[t];
    var vector = I.toVector(state.analysis.features);
    var list = clear($("tree-path"));
    var node = tree.root, step = 0;
    while (tree.f[node] !== -1) {
      var name = I.FEATURES[tree.f[node]];
      var v = vector[tree.f[node]];
      var goYes = Math.fround(v) < Math.fround(tree.t[node]);
      list.appendChild(el("li", { class: goYes ? "yes" : "no", style: "animation-delay:" + (step * 0.08) + "s",
        text: name + " = " + (Math.abs(v - Math.round(v)) < 1e-9 ? v : v.toFixed(3)) + (goYes ? "  <  " : "  >=  ") + tree.t[node] + "   → " + (goYes ? "yes branch" : "no branch") }));
      node = goYes ? tree.y[node] : tree.n[node];
      step++;
    }
    var leaf = tree.t[node];
    list.appendChild(el("li", { class: "leaf", style: "animation-delay:" + (step * 0.08) + "s",
      text: "leaf value " + (leaf >= 0 ? "+" : "") + leaf.toFixed(5) + "  (already multiplied by the learning rate)" }));
  }

  // --------------------------------------------------------------------------------------
  // Step 6: held-out scatter and Spearman demo
  // --------------------------------------------------------------------------------------
  function ranks(values) {
    var order = values.map(function (v, i) { return [v, i]; }).sort(function (a, b) { return a[0] - b[0]; });
    var r = new Array(values.length);
    var i = 0;
    while (i < order.length) {
      var j = i;
      while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
      var avg = (i + j) / 2 + 1;
      for (var k = i; k <= j; k++) r[order[k][1]] = avg;
      i = j + 1;
    }
    return r;
  }

  function pearson(a, b) {
    var n = a.length, ma = 0, mb = 0;
    for (var i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    var num = 0, da = 0, db = 0;
    for (i = 0; i < n; i++) { num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) * (a[i] - ma); db += (b[i] - mb) * (b[i] - mb); }
    return num / Math.sqrt(da * db);
  }

  function renderScatter() {
    var W = 520, H = 330, left = 44, bottom = 64, top = 10;
    var x = scale(-0.5, 12.5, left, W - 10), y = scale(0.2, 1.15, H - bottom, top);
    var svg = chart(W, H);
    [0.3, 0.5, 0.7, 0.9, 1.1].forEach(function (s) {
      svg.appendChild(svgEl("line", { x1: left, x2: W - 10, y1: y(s), y2: y(s), class: "grid" }));
      svg.appendChild(svgEl("text", { x: left - 6, y: y(s) + 4, "text-anchor": "end", text: s.toFixed(1) }));
    });
    var rand = mulberry32(5);
    var byTier = {};
    D.scatter.forEach(function (pt, i) {
      (byTier[pt[0]] = byTier[pt[0]] || []).push(pt[1]);
      var c = svgEl("circle", { cx: x(pt[0] + (rand() - 0.5) * 0.64), cy: y(pt[1]), r: 2.4, fill: css("--cyan"), opacity: 0 });
      c.style.transition = "opacity 0.5s " + (i * 1.5) + "ms";
      svg.appendChild(c);
      requestAnimationFrame(function () { c.setAttribute("opacity", 0.5); });
    });
    var med = Object.keys(byTier).map(Number).sort(function (a, b) { return a - b; }).map(function (t) {
      var v = byTier[t].slice().sort(function (a, b) { return a - b; });
      return [t, v[Math.floor(v.length / 2)]];
    });
    svg.appendChild(svgEl("path", { d: med.map(function (m, i) { return (i ? "L" : "M") + x(m[0]) + "," + y(m[1]); }).join(" "), fill: "none", stroke: css("--amber"), "stroke-width": 2.5 }));
    D.owners_tiers.forEach(function (t, i) {
      var tx = x(i), ty = H - bottom + 12;
      svg.appendChild(svgEl("text", { x: tx, y: ty, "text-anchor": "end", transform: "rotate(-40 " + tx + " " + ty + ")", class: "mono", text: t.label }));
    });
    clear($("chart-scatter")).appendChild(svg);
  }

  function renderSpearmanDemo() {
    var power = parseInt($("in-pow").value, 10);
    var noise = parseFloat($("in-noise").value);
    $("pow-out").textContent = power;
    $("noise-out").textContent = noise.toFixed(2);
    var rand = mulberry32(17);
    var tiers = D.scatter.map(function (p) { return p[0]; });
    var scores = D.scatter.map(function (p) { return Math.pow(Math.max(1e-6, p[1] + noise * gaussian(rand)), power); });
    $("rho-s").textContent = pearson(ranks(tiers), ranks(scores)).toFixed(3);
    $("rho-p").textContent = pearson(tiers, scores).toFixed(3);
  }

  // --------------------------------------------------------------------------------------
  // Step 7: CDF of the reference distribution
  // --------------------------------------------------------------------------------------
  function renderCdf() {
    var ref = M.reference_ranks, n = ref.length;
    var a = state.analysis;
    var W = 900, H = 320, left = 48, bottom = 36, top = 12;
    var smin = ref[0] - 0.02, smax = ref[n - 1] + 0.02;
    var x = scale(smin, smax, left, W - 12), y = scale(0, 100, H - bottom, top);
    var svg = chart(W, H);
    svg.appendChild(svgEl("rect", { x: left, y: y(1), width: W - 12 - left, height: y(0) - y(1), fill: css("--red"), opacity: 0.15 }));
    svg.appendChild(svgEl("rect", { x: left, y: y(100), width: W - 12 - left, height: y(99) - y(100), fill: css("--red"), opacity: 0.15 }));
    [0, 25, 50, 75, 100].forEach(function (p) {
      svg.appendChild(svgEl("line", { x1: left, x2: W - 12, y1: y(p), y2: y(p), class: "grid" }));
      svg.appendChild(svgEl("text", { x: left - 6, y: y(p) + 4, "text-anchor": "end", text: String(p) }));
    });
    [0.3, 0.5, 0.7, 0.9, 1.1].forEach(function (s) { svg.appendChild(svgEl("text", { x: x(s), y: H - 14, "text-anchor": "middle", class: "mono", text: s.toFixed(1) })); });
    var pts = [];
    for (var i = 0; i < n; i += 15) pts.push([ref[i], i / n * 100]);
    pts.push([ref[n - 1], 100]);
    svg.appendChild(svgEl("path", { d: pts.map(function (p, k) { return (k ? "L" : "M") + x(p[0]).toFixed(1) + "," + y(p[1]).toFixed(1); }).join(" "), fill: "none", stroke: css("--violet"), "stroke-width": 3 }));

    var below = 0;
    while (below < n && ref[below] < a.score) below++;
    var raw = below / n * 100;
    var sx = x(Math.max(smin, Math.min(smax, a.score))), sy = y(raw);
    var guide = svgEl("path", { d: "M" + sx + "," + (H - bottom) + " L" + sx + "," + sy + " L" + left + "," + sy, fill: "none", stroke: css("--cyan"), "stroke-width": 2, "stroke-dasharray": "5 4" });
    svg.appendChild(guide);
    svg.appendChild(svgEl("circle", { cx: sx, cy: sy, r: 6, fill: css("--cyan") }));
    svg.appendChild(svgEl("text", { x: left + 8, y: sy - 8, class: "t-strong", text: ordinal(a.percentile) + " percentile" }));
    svg.appendChild(svgEl("text", { x: sx + 8, y: H - bottom - 8, class: "mono", text: "score " + a.score.toFixed(4) }));
    clear($("chart-cdf")).appendChild(svg);
    $("cdf-score").textContent = a.score.toFixed(4);
    $("cdf-below").textContent = fmtInt(below) + " / " + fmtInt(n);
    $("cdf-pct").textContent = ordinal(a.percentile);
  }

  // --------------------------------------------------------------------------------------
  // Step 8: what-if engine
  // --------------------------------------------------------------------------------------
  // How each candidate can be reproduced through the Streamlit form (null: impossible).
  var FORM_ACTIONS = {
    platform_count: function (g) { g.platforms = ["windows", "mac", "linux"]; },
    has_achievements: function (g) { g.achievements = Math.max(1, g.achievements); toggleInList(g.categories, "Steam Achievements", true); },
    achievements: function (g) { g.achievements = 30; },
    cat_steam_trading_cards: function (g) { toggleInList(g.categories, "Steam Trading Cards", true); },
    cat_steam_cloud: function (g) { toggleInList(g.categories, "Steam Cloud", true); },
    cat_multi_player: null,
    cat_co_op: function (g) { toggleInList(g.categories, "Co-op", true); },
    cat_online_co_op: function (g) { toggleInList(g.categories, "Online Co-op", true); },
    cat_full_controller_support: function (g) { toggleInList(g.categories, "Full controller support", true); },
    cat_steam_workshop: function (g) { toggleInList(g.categories, "Steam Workshop", true); },
    tag_multiplayer: function (g) { toggleInList(g.tags, "Multiplayer", true); },
    tag_co_op: function (g) { toggleInList(g.tags, "Co-op", true); },
    tag_story_rich: function (g) { toggleInList(g.tags, "Story Rich", true); }
  };

  function renderWhatIf() {
    var a = state.analysis;
    var table = clear($("whatif-table"));
    table.appendChild(el("tr", {}, ["candidate change", "status", "gain (percentile points)", "", "confidence", ""].map(function (h) { return el("th", { text: h }); })));
    var topSet = {};
    a.improvements.forEach(function (imp, i) { topSet[imp.feature] = i + 1; });
    var maxGain = Math.max(5, Math.max.apply(null, a.evaluated.map(function (e) { return e.gain || 0; })));
    var rows = a.evaluated.slice().sort(function (p, q) { return (q.gain === null ? -1 : q.gain) - (p.gain === null ? -1 : p.gain); });

    rows.forEach(function (ev) {
      var kept = topSet[ev.feature];
      var cls, status;
      if (kept) { cls = "kept"; status = "#" + kept + " shown by the app"; }
      else if (ev.status === "kept") { cls = "cut"; status = "kept, beyond top 6"; }
      else if (ev.gain !== null) { cls = "drop"; status = ev.status; }
      else { cls = "skip"; status = "skipped: " + ev.status; }
      var validation = M.validated_improvements[ev.feature];
      var conf = validation ? validation.confidence : "Medium";
      var bar = el("div", { class: "gain-bar" });
      var action = FORM_ACTIONS[ev.feature];
      var applyBtn = el("button", { class: "btn small", type: "button", text: "apply" });
      if (!action || ev.gain === null) {
        applyBtn.disabled = true;
        applyBtn.title = action === null ? "The Streamlit form has no 'Multi-player' checkbox" : "Not applicable";
      } else {
        applyBtn.addEventListener("click", function () {
          var predicted = I.getPercentile(Object.assign({}, a.features, (function () { var o = {}; o[ev.feature] = I.IMPROVEMENT_OPTIONS.filter(function (opt) { return opt[0] === ev.feature; })[0][1]; return o; })()));
          action(state.game);
          state.applyNote = { description: ev.description, predicted: predicted, before: a.percentile };
          update();
        });
      }
      table.appendChild(el("tr", { class: cls }, [
        el("td", { text: ev.description }),
        el("td", { class: "source", text: status }),
        el("td", { class: "bar-cell" }, [bar]),
        el("td", { class: "mono", text: ev.gain === null ? "" : (ev.gain >= 0 ? "+" : "") + ev.gain.toFixed(2) }),
        el("td", {}, [ev.gain !== null && ev.gain > 0.5 ? el("span", { class: "badge " + conf, text: conf }) : null]),
        el("td", {}, [applyBtn])
      ]));
      requestAnimationFrame(function () { bar.style.width = (ev.gain ? Math.max(0, ev.gain) / maxGain * 100 : 0) + "%"; });
    });

    var note = $("apply-note");
    if (state.applyNote) {
      var n = state.applyNote;
      var diff = Math.abs(n.predicted - a.percentile);
      note.style.display = "block";
      note.innerHTML = "Applied <b>" + n.description + "</b>. The what-if engine predicted <b>" + ordinal(n.predicted) +
        "</b>; re-analysing the modified form gives <b>" + ordinal(a.percentile) + "</b>" +
        (diff > 0.05 ? ". The two differ because the what-if changes a single column, while the form also updates the related columns " +
          "(for example <code>mac</code> and <code>linux</code> together with <code>platform_count</code>)." : ", identical as expected.");
    } else {
      note.style.display = "none";
    }
  }

  function renderValidationTable() {
    var table = $("validation-table");
    table.appendChild(el("tr", {}, ["feature", "games with", "games without", "median owners lift", "confidence"].map(function (h) { return el("th", { text: h }); })));
    ["platform_count", "has_achievements", "cat_steam_trading_cards", "cat_steam_cloud", "cat_multi_player", "tag_multiplayer", "tag_co_op"].forEach(function (f) {
      var v = M.validated_improvements[f];
      if (v) {
        table.appendChild(el("tr", {}, [
          el("td", { class: "mono", text: f }), el("td", { class: "num", text: fmtInt(v.n_with) }), el("td", { class: "num", text: fmtInt(v.n_without) }),
          el("td", { class: "num", text: (v.actual_lift > 0 ? "+" : "") + v.actual_lift.toFixed(0) + "%" }),
          el("td", {}, [el("span", { class: "badge " + v.confidence, text: v.confidence })])
        ]));
      } else {
        var reason = f === "platform_count"
          ? "not validated: the check compares platform_count == 1 against == 0, and no test game has 0 platforms"
          : "not validated: only 31 test games carry the tag (minimum is 50)";
        table.appendChild(el("tr", {}, [el("td", { class: "mono", text: f }), el("td", { colspan: 4, class: "source", text: reason + "; shown as Medium" })]));
      }
    });
  }

  // --------------------------------------------------------------------------------------
  // Step 9: dashboard (mirror of new_game.display_analysis_results)
  // --------------------------------------------------------------------------------------
  var gaugeNeedle = null;

  function buildGauge() {
    var W = 420, H = 250, cx = 210, cy = 200, r = 150;
    var svg = chart(W, H);
    var bands = [[0, 20, "#dc3545"], [20, 40, "#fd7e14"], [40, 60, "#ffc107"], [60, 80, "#17a2b8"], [80, 100, "#28a745"]];
    function pt(p, rad) { var th = Math.PI * (1 - p / 100); return [cx + rad * Math.cos(th), cy - rad * Math.sin(th)]; }
    bands.forEach(function (b) {
      var p1 = pt(b[0] + 0.5, r), p2 = pt(b[1] - 0.5, r);
      svg.appendChild(svgEl("path", { d: "M" + p1[0] + "," + p1[1] + " A" + r + "," + r + " 0 0 1 " + p2[0] + "," + p2[1], stroke: b[2], "stroke-width": 26, fill: "none" }));
    });
    [0, 20, 40, 60, 80, 100].forEach(function (p) { var q = pt(p, r - 30); svg.appendChild(svgEl("text", { x: q[0], y: q[1] + 4, "text-anchor": "middle", text: String(p) })); });
    gaugeNeedle = svgEl("g", { style: "transform-origin: " + cx + "px " + cy + "px; transition: transform 0.9s cubic-bezier(.3,1.4,.5,1); transform: rotate(-90deg)" }, [
      svgEl("line", { x1: cx, y1: cy, x2: cx, y2: cy - r + 30, stroke: css("--text"), "stroke-width": 4, "stroke-linecap": "round" })
    ]);
    svg.appendChild(gaugeNeedle);
    svg.appendChild(svgEl("circle", { cx: cx, cy: cy, r: 9, fill: css("--text") }));
    svg.appendChild(svgEl("text", { x: cx, y: cy + 40, "text-anchor": "middle", class: "t-strong", id: "gauge-label", style: "font-size:22px" }));
    $("gauge").appendChild(svg);
  }

  function renderDashboard() {
    var a = state.analysis;
    var p = a.percentile;
    if (gaugeNeedle) gaugeNeedle.style.transform = "rotate(" + (-90 + p * 1.8) + "deg)";
    var label = document.getElementById("gauge-label");
    if (label) label.textContent = p.toFixed(1) + " market percentile";

    var tier = a.tier.replace("_", " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
    var metrics = clear($("dash-metrics"));
    [[p.toFixed(0) + "th", p >= 50 ? "Top " + (100 - p).toFixed(0) + "%" : "Bottom " + p.toFixed(0) + "%", "Market Percentile"],
     [tier, "", "Success Tier"],
     [a.success_factors.readiness_score + "%", "max 70% by construction", "Readiness Score"]].forEach(function (m) {
      metrics.appendChild(el("div", { class: "stat" }, [el("div", { class: "l", text: m[2] }), el("div", { class: "v", text: m[0] }), el("div", { class: "l", text: m[1] })]));
    });

    var ins = clear($("dash-insights"));
    a.insights.forEach(function (x) { ins.appendChild(el("div", { class: "insight " + x.type }, [el("b", { text: x.title }), el("div", { class: "muted", text: x.message })])); });

    var risks = clear($("dash-risks"));
    if (!a.risks.length) risks.appendChild(el("div", { class: "callout ok", text: "No significant risks identified with this configuration." }));
    a.risks.forEach(function (r) {
      risks.appendChild(el("div", { class: "risk" }, [el("span", { class: "lvl " + r.level, text: r.level }),
        el("div", {}, [el("b", { text: r.title }), el("div", { class: "muted", text: r.description + " Recommendation: " + r.recommendation })])]));
    });

    var sf = a.success_factors;
    var factors = clear($("dash-factors"));
    function list(title, items) {
      return el("div", {}, [el("div", { class: "fam-title", text: title }), el("ul", { class: "factors" }, items.map(function (f) {
        return el("li", { text: f.factor + " (" + f.impact + " impact, +" + (f.impact === "high" ? 30 : 20) + ")" });
      }))]);
    }
    factors.appendChild(el("div", { class: "grid-2" }, [list("Present", sf.present), list("Missing", sf.missing)]));
  }

  // --------------------------------------------------------------------------------------
  // Step 10: validation
  // --------------------------------------------------------------------------------------
  function renderValidation() {
    var cards = [
      [M.spearman.toFixed(3), "Ranker Spearman, random split (" + fmtInt(M.n_test) + " test games, 95% CI 0.712 to 0.738)"],
      ["0.432", "Same ranker recipe on a temporal split (train before 2018, test from 2018)"],
      ["0.362", "Owners model R² on log owners (temporal split; Spearman 0.429)"],
      ["0.110", "Review-ratio model R² (MAE 0.197)"]
    ];
    var host = $("metric-cards");
    cards.forEach(function (c) { host.appendChild(el("div", { class: "stat" }, [el("div", { class: "v", text: c[0] }), el("div", { class: "l", text: c[1] })])); });

    var table = $("sample-table");
    table.appendChild(el("tr", {}, ["game (test split)", "release", "price", "owners range", "predicted percentile", "actual percentile"].map(function (h) { return el("th", { text: h }); })));
    D.sample_games.forEach(function (g) {
      table.appendChild(el("tr", {}, [
        el("td", { text: g.name }), el("td", { class: "num", text: String(g.release_year) }), el("td", { class: "num", text: "$" + g.price.toFixed(2) }),
        el("td", { class: "mono", text: g.owners }), el("td", { class: "num", text: g.predicted_percentile.toFixed(1) }), el("td", { class: "num", text: g.actual_percentile.toFixed(1) })
      ]));
    });
  }

  // --------------------------------------------------------------------------------------
  // Navigation and theme
  // --------------------------------------------------------------------------------------
  function bindToc() {
    var links = document.querySelectorAll(".toc a");
    if (!("IntersectionObserver" in window)) return;
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        links.forEach(function (a) { a.classList.toggle("active", a.getAttribute("href") === "#" + entry.target.id); });
      });
    }, { rootMargin: "-30% 0px -60% 0px" });
    document.querySelectorAll("main section").forEach(function (s) { observer.observe(s); });
  }

  function bindTheme() {
    var stored = null;
    try { stored = localStorage.getItem("idss-theme"); } catch (e) { stored = null; }
    if (stored) document.documentElement.setAttribute("data-theme", stored);
    $("theme-toggle").addEventListener("click", function () {
      var next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("idss-theme", next); } catch (e) { /* storage unavailable */ }
      redrawAll();
    });
  }

  var ownersMode = "log", splitMode = "random";

  function redrawAll() {
    renderOwnersChart(ownersMode);
    renderYearsChart(splitMode);
    renderToy();
    renderScatter();
    clear($("gauge"));
    buildGauge();
    update();
  }

  function init() {
    bindTheme();
    renderStages();
    renderDataStats();
    renderOwnersChart(ownersMode);
    renderYearsChart(splitMode);
    bindSeg("owners-scale", function (v) { ownersMode = v; renderOwnersChart(v); });
    bindSeg("split-mode", function (v) { splitMode = v; renderYearsChart(v); });
    buildForm();
    bindToy();
    $("tree-idx").addEventListener("input", renderTreePath);
    renderScatter();
    renderSpearmanDemo();
    $("in-pow").addEventListener("input", renderSpearmanDemo);
    $("in-noise").addEventListener("input", renderSpearmanDemo);
    renderValidationTable();
    buildGauge();
    renderValidation();
    bindToc();
    update();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
