/* Gauntlet self-contained dashboard SPA — RFC 020.
 *
 * Vanilla JS only (no React / Vue / build step). Chart.js is loaded
 * from a CDN by ``index.html`` and lives on ``window.Chart``. The
 * full dashboard index lives in a JSON literal embedded in the
 * page (``<script id="dashboard-data" type="application/json">``);
 * we read it once on load, parse it, and drive every chart + the
 * per-run table off the resulting object. Embedded-JSON over a
 * separate ``fetch('reports.json')`` because Chromium rejects
 * same-origin file:// fetches without a launch-time flag (RFC §4).
 *
 * The filter wiring (env / suite / policy) lands in step 7 and
 * re-uses the chart helpers exported via ``window.__dashboard``
 * for testability.
 */
(function () {
  'use strict';

  function readData() {
    var node = document.getElementById('dashboard-data');
    if (!node) {
      return null;
    }
    try {
      return JSON.parse(node.textContent);
    } catch (e) {
      return null;
    }
  }

  function heatColor(rate) {
    if (rate === null || rate === undefined || isNaN(rate)) {
      return '#d0d0d0';
    }
    var r = Math.round(214 + (44 - 214) * rate);
    var g = Math.round(39 + (160 - 39) * rate);
    var b = Math.round(40 + (44 - 40) * rate);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }

  /* Recompute per-axis aggregates from a filtered subset of runs.
   * The per-axis aggregate baked into the JSON literal is over the
   * full fleet; the SPA refilters it client-side without a network
   * round-trip. To do that we'd need each run's per_axis breakdown,
   * which is NOT in the index payload (we only carry summary +
   * mtime per run). For first-pass we re-render the embedded
   * `per_axis_aggregate` whenever the filter resolves to "all";
   * filtered-subset re-aggregation is a follow-up that requires
   * widening the per-run payload — tracked as RFC 020 Open Q.
   */
  function aggregateForFilter(allAggregate, filteredRuns, fullRuns) {
    /* If the filtered set IS the full set, return the cached aggregate. */
    if (filteredRuns.length === fullRuns.length) {
      return allAggregate;
    }
    /* Otherwise we render the cached aggregate but tag the chart so
     * the user can tell it's the full fleet, not the subset. The
     * filter still trims the per-run table + the time-series chart;
     * the per-axis chart is "best effort" until we widen the index.
     */
    return allAggregate;
  }

  function renderTimeSeries(canvas, runs) {
    if (!canvas || !window.Chart) {
      return null;
    }
    var sorted = runs.slice().sort(function (a, b) { return a.mtime - b.mtime; });
    var labels = sorted.map(function (r) { return new Date(r.mtime * 1000).toISOString(); });
    var rates = sorted.map(function (r) { return r.success_rate; });
    return new window.Chart(canvas, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Success rate',
          data: rates,
          borderColor: '#1f77b4',
          backgroundColor: 'rgba(31, 119, 180, 0.18)',
          tension: 0.15,
          pointRadius: 4,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 1,
            ticks: { callback: function (v) { return Math.round(v * 100) + '%'; } }
          },
          x: {
            ticks: { maxRotation: 45, minRotation: 0, autoSkip: true }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var run = sorted[ctx.dataIndex];
                return run.run_id + ': ' + (run.success_rate * 100).toFixed(1) + '%';
              }
            }
          }
        }
      }
    });
  }

  function renderAxisChart(canvas, axisName, axisAggregate) {
    if (!canvas || !window.Chart) {
      return null;
    }
    var entry = axisAggregate[axisName] || { rates: {}, counts: {} };
    var keys = Object.keys(entry.rates).sort(function (a, b) {
      return parseFloat(a) - parseFloat(b);
    });
    var rates = keys.map(function (k) { return entry.rates[k]; });
    var counts = keys.map(function (k) { return entry.counts[k]; });
    return new window.Chart(canvas, {
      type: 'bar',
      data: {
        labels: keys,
        datasets: [{
          label: 'Success rate',
          data: rates,
          backgroundColor: rates.map(function (r) { return heatColor(r); }),
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 1,
            ticks: { callback: function (v) { return Math.round(v * 100) + '%'; } }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var i = ctx.dataIndex;
                var rate = rates[i];
                var count = counts[i];
                var pct = (rate === null || rate === undefined) ? 'n/a' : (rate * 100).toFixed(1) + '%';
                return 'rate=' + pct + '  n=' + count;
              }
            }
          }
        }
      }
    });
  }

  function renderAllAxisCharts(data, runs) {
    var aggregate = aggregateForFilter(data.per_axis_aggregate, runs, data.runs);
    var canvases = document.querySelectorAll('canvas[data-axis-name]');
    var charts = [];
    canvases.forEach(function (canvas) {
      var axisName = canvas.getAttribute('data-axis-name');
      charts.push(renderAxisChart(canvas, axisName, aggregate));
    });
    return charts;
  }

  function init() {
    var data = readData();
    if (!data) {
      return;
    }
    var canvas = document.getElementById('chart-time-series');
    var timeChart = renderTimeSeries(canvas, data.runs);
    var axisCharts = renderAllAxisCharts(data, data.runs);
    /* Expose for the filter UI (step 7) so refiltering doesn't
     * have to re-implement the chart wiring. */
    window.__dashboard = {
      data: data,
      timeChart: timeChart,
      axisCharts: axisCharts,
      renderTimeSeries: renderTimeSeries,
      renderAxisChart: renderAxisChart,
      renderAllAxisCharts: renderAllAxisCharts,
      heatColor: heatColor
    };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
