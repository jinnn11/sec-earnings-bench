// The results explorer's script (secbench.hfsite's pages; copied into the Space's
// assets/ by secbench.hfspace). It shows one split at a time and hides the rows a
// filter leaves out. The pages hold every row, and what each filter matches is
// computed by the build (a release list's data in its <script class="filter-data">),
// so the script only looks things up; without it the pages show everything.
(function () {
  "use strict";

  document.documentElement.classList.add("js");

  function all(root, selector) {
    return Array.prototype.slice.call(root.querySelectorAll(selector));
  }

  // The split a URL names: explore.html#split=time_test.
  function splitInUrl() {
    var match = /^#split=([A-Za-z0-9_-]+)$/.exec(window.location.hash);
    return match ? match[1] : null;
  }

  function setUpSplits() {
    var bar = document.querySelector(".split-tabs");
    if (!bar) {
      return;
    }
    var buttons = all(bar, "button[data-split]");
    var sections = all(document, "section.split[data-split]");
    function show(name, remember) {
      var known = sections.some(function (section) {
        return section.getAttribute("data-split") === name;
      });
      if (!known) {
        name = bar.getAttribute("data-default");
      }
      sections.forEach(function (section) {
        section.classList.toggle("current", section.getAttribute("data-split") === name);
      });
      buttons.forEach(function (button) {
        var chosen = button.getAttribute("data-split") === name;
        button.setAttribute("aria-selected", chosen ? "true" : "false");
      });
      if (remember && window.history && window.history.replaceState) {
        window.history.replaceState(null, "", "#split=" + name);
      }
    }
    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        show(button.getAttribute("data-split"), true);
      });
    });
    window.addEventListener("hashchange", function () {
      show(splitInUrl(), false);
    });
    show(splitInUrl(), false);
  }

  // Explore: the releases where a model (or any) has a figure with a verdict (or
  // any), as the build listed them, and whose ticker or name holds the search.
  function setUpReleaseFilters(section) {
    var holder = section.querySelector("script.filter-data");
    var filters = section.querySelector(".filters");
    var model = section.querySelector('select[data-filter="model"]');
    var verdict = section.querySelector('select[data-filter="verdict"]');
    var company = section.querySelector('input[data-filter="company"]');
    var status = section.querySelector(".status");
    if (!holder || !filters || !model || !verdict || !company || !status) {
      return;
    }
    var matches = JSON.parse(holder.textContent);
    var rows = all(section, "tr[data-i]");
    function apply() {
      var m = model.value;
      var v = verdict.value;
      var q = company.value.trim().toLowerCase();
      var keep = null;
      if (m || v) {
        keep = {};
        ((matches[m] || {})[v] || []).forEach(function (i) {
          keep[i] = true;
        });
      }
      var shown = 0;
      var companies = {};
      rows.forEach(function (row) {
        var ok =
          (keep === null || keep[row.getAttribute("data-i")] === true) &&
          (!q || row.getAttribute("data-search").indexOf(q) !== -1);
        row.hidden = !ok;
        if (ok) {
          shown += 1;
          companies[row.getAttribute("data-ticker")] = true;
        }
      });
      var n = Object.keys(companies).length;
      if (!m && !v && !q) {
        status.textContent = status.getAttribute("data-default");
      } else {
        status.textContent =
          shown + " of the " + rows.length + " releases match, from " + n +
          (n === 1 ? " company." : " companies.");
      }
    }
    model.addEventListener("change", function () {
      // Picking a model with no verdict chosen shows that model's misses.
      if (model.value && !verdict.value) {
        verdict.value = filters.getAttribute("data-miss");
      }
      apply();
    });
    verdict.addEventListener("change", apply);
    company.addEventListener("input", apply);
    apply();
  }

  // Errors: the examples of one model (or any) with one verdict (or any miss).
  function setUpExampleFilters(section) {
    var filters = section.querySelector(".filters");
    var model = section.querySelector('select[data-filter="example-model"]');
    var verdict = section.querySelector('select[data-filter="example-verdict"]');
    var empty = section.querySelector(".empty");
    if (!filters || !model || !verdict) {
      return;
    }
    var miss = filters.getAttribute("data-miss");
    var rows = all(section, "tr[data-run]");
    function apply() {
      var m = model.value;
      var v = verdict.value;
      var shown = 0;
      rows.forEach(function (row) {
        var ok =
          (!m || row.getAttribute("data-run") === m) &&
          (!v ||
            (v === miss
              ? row.getAttribute("data-miss") === "1"
              : row.getAttribute("data-verdict") === v));
        row.hidden = !ok;
        if (ok) {
          shown += 1;
        }
      });
      if (empty) {
        empty.hidden = shown > 0;
      }
    }
    model.addEventListener("change", apply);
    verdict.addEventListener("change", apply);
    apply();
  }

  function start() {
    setUpSplits();
    all(document, "section.split").forEach(function (section) {
      setUpReleaseFilters(section);
      setUpExampleFilters(section);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
