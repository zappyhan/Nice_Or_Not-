/* Owner-facing dashboard.
 *
 * Plain ES2020 -- no framework, no bundler, no CDN -- so the page works offline
 * inside the container and cannot rot when a CDN version moves. It reads the
 * same JSON endpoints the batch pipeline writes, so the screen and the report
 * can never disagree.
 *
 * Charting is hand-rolled SVG. Three rules are applied throughout:
 *   - one data hue for every mark; colour never encodes rank
 *   - status (risk band, alert severity) always ships a dot AND a written label
 *   - every chart has a table view, so nothing is readable by colour alone
 */

const ASPECT_NAMES = {
  service: "Service",
  food_quality: "Food quality",
  cleanliness: "Cleanliness",
  price_value: "Price / value",
  ambience: "Ambience",
  wait_time: "Wait time",
};

const state = {
  venues: [],
  businessId: null,
  report: null,
  history: null,
  aspects: null,
  alerts: [],
  plan: null,
  selectedAspect: null,
};

// ---------------------------------------------------------------- utilities

const $ = (id) => document.getElementById(id);
const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;
const pct1 = (v) => `${((v ?? 0) * 100).toFixed(1)}%`;
const stars = (v) => (v == null ? "–" : v.toFixed(2));

/** Month key "2024-05" -> "May 2024" for humans. */
function monthLabel(key) {
  const [y, m] = key.split("-").map(Number);
  const names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${names[m - 1]} ${y}`;
}

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText} for ${url}`);
  return response.json();
}

// ------------------------------------------------------------------ tooltip

const tooltip = $("tooltip");

function showTooltip(event, title, rows) {
  tooltip.innerHTML =
    `<div class="tooltip-title">${title}</div>` +
    rows.map((r) => `<div class="tooltip-row">${r}</div>`).join("");
  tooltip.hidden = false;
  // Clamp to the viewport so a tooltip near the right edge stays readable.
  const box = tooltip.getBoundingClientRect();
  const x = Math.min(event.clientX + 14, window.innerWidth - box.width - 10);
  const y = Math.max(10, event.clientY - box.height - 12);
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

const hideTooltip = () => { tooltip.hidden = true; };

// ------------------------------------------------------------- SVG helpers

const SVG_NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

/** Rect with only its data end rounded, anchored to the baseline at x = 0. */
function rightRoundedBar(x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius, width, height / 2));
  return `M${x},${y} H${x + width - r} A${r},${r} 0 0 1 ${x + width},${y + r}` +
         ` V${y + height - r} A${r},${r} 0 0 1 ${x + width - r},${y + height} H${x} Z`;
}

/** Tick values a human would choose: 1, 2, 2.5 or 5 times a power of ten. */
function niceTicks(min, max, count) {
  const raw = (max - min) / count || 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const normalised = raw / magnitude;
  const step = (normalised <= 1 ? 1 : normalised <= 2 ? 2 : normalised <= 2.5 ? 2.5 : normalised <= 5 ? 5 : 10) * magnitude;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) {
    out.push(Number(v.toFixed(10)));
  }
  return out;
}

/** Round a maximum up to the next nice step, so the axis ends on a round number. */
function niceCeiling(value, count = 4) {
  const t = niceTicks(0, value, count);
  const step = t.length > 1 ? t[1] - t[0] : value;
  return Math.ceil(value / step) * step;
}

// -------------------------------------------------------------- line chart

/**
 * Single-series line chart with a crosshair tooltip.
 * Null values break the path rather than interpolating across them, so a month
 * with no reviews reads as a gap instead of an invented value.
 */
function lineChart(container, options) {
  const { points, yDomain, format, area = false, yLabel = "" } = options;
  container.innerHTML = "";

  const width = Math.max(container.clientWidth || 520, 280);
  const height = 250;
  const margin = { top: 14, right: 18, bottom: 30, left: 46 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height });
  const root = el("g", { transform: `translate(${margin.left},${margin.top})` });
  svg.appendChild(root);

  const [yMin, yMax] = yDomain;
  const xAt = (i) => (points.length <= 1 ? plotW / 2 : (i / (points.length - 1)) * plotW);
  const yAt = (v) => plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;

  // Gridlines and y axis (recessive -- they orient, they do not compete).
  for (const t of niceTicks(yMin, yMax, 4)) {
    const y = yAt(t);
    root.appendChild(el("line", { class: "grid-line", x1: 0, y1: y, x2: plotW, y2: y }));
    const label = el("text", { class: "axis-text", x: -8, y: y + 4, "text-anchor": "end" });
    label.textContent = format(t);
    root.appendChild(label);
  }
  root.appendChild(el("line", { class: "axis-line", x1: 0, y1: plotH, x2: plotW, y2: plotH }));

  // x labels: as many as fit without colliding. A month label needs ~78px, so
  // a narrow phone gets three labels and a wide desktop gets six -- never a
  // label per month, and never two labels on top of each other.
  const maxLabels = Math.max(2, Math.floor(plotW / 78));
  const stride = Math.max(1, Math.ceil((points.length - 1) / maxLabels));
  const labelled = [];
  for (let i = 0; i < points.length; i += stride) labelled.push(i);
  const last = points.length - 1;
  // Keep the final month, but drop the one before it if they would overlap.
  if (labelled[labelled.length - 1] !== last) {
    if (last - labelled[labelled.length - 1] < stride * 0.6) labelled.pop();
    labelled.push(last);
  }
  for (const i of labelled) {
    const label = el("text", { class: "axis-text", x: xAt(i), y: plotH + 18, "text-anchor": "middle" });
    label.textContent = monthLabel(points[i].key);
    root.appendChild(label);
  }

  // Split into contiguous runs of non-null values.
  const runs = [];
  let run = [];
  points.forEach((point, i) => {
    if (point.value == null) {
      if (run.length) runs.push(run);
      run = [];
    } else {
      run.push({ x: xAt(i), y: yAt(point.value) });
    }
  });
  if (run.length) runs.push(run);

  for (const segment of runs) {
    const d = segment.map((p, i) => `${i ? "L" : "M"}${p.x},${p.y}`).join(" ");
    if (area && segment.length > 1) {
      const last = segment[segment.length - 1];
      root.appendChild(el("path", {
        class: "series-area",
        d: `${d} L${last.x},${plotH} L${segment[0].x},${plotH} Z`,
      }));
    }
    root.appendChild(el("path", { class: "series-line", d }));
    if (segment.length === 1) {
      root.appendChild(el("circle", { class: "series-dot", cx: segment[0].x, cy: segment[0].y, r: 4 }));
    }
  }

  // Hover layer: crosshair + marker on the nearest month.
  const crosshair = el("line", { class: "crosshair", y1: 0, y2: plotH, opacity: 0 });
  const marker = el("circle", { class: "series-dot", r: 5, opacity: 0 });
  root.appendChild(crosshair);
  root.appendChild(marker);

  const overlay = el("rect", { x: 0, y: 0, width: plotW, height: plotH, fill: "transparent" });
  overlay.addEventListener("mousemove", (event) => {
    const bounds = svg.getBoundingClientRect();
    const scale = plotW / (bounds.width - margin.left - margin.right || 1);
    const localX = (event.clientX - bounds.left - margin.left) * scale;
    let nearest = 0;
    points.forEach((_, i) => {
      if (Math.abs(xAt(i) - localX) < Math.abs(xAt(nearest) - localX)) nearest = i;
    });
    const point = points[nearest];
    crosshair.setAttribute("x1", xAt(nearest));
    crosshair.setAttribute("x2", xAt(nearest));
    crosshair.setAttribute("opacity", 1);
    if (point.value == null) {
      marker.setAttribute("opacity", 0);
      showTooltip(event, monthLabel(point.key), ["No reviews this month"]);
    } else {
      marker.setAttribute("cx", xAt(nearest));
      marker.setAttribute("cy", yAt(point.value));
      marker.setAttribute("opacity", 1);
      showTooltip(event, monthLabel(point.key), [
        `${yLabel}: <strong>${format(point.value)}</strong>`,
        `${point.n_reviews ?? 0} review${point.n_reviews === 1 ? "" : "s"}`,
      ]);
    }
  });
  overlay.addEventListener("mouseleave", () => {
    crosshair.setAttribute("opacity", 0);
    marker.setAttribute("opacity", 0);
    hideTooltip();
  });
  root.appendChild(overlay);

  container.appendChild(svg);
}

// --------------------------------------------------------- horizontal bars

/**
 * Complaint rate per issue, with the market median drawn as a reference tick.
 * The median is chrome, not a second series: it is a benchmark line, so it gets
 * a neutral tick and a key entry rather than a competing colour.
 */
function barChart(container, options) {
  const { items, selected, onSelect } = options;
  container.innerHTML = "";

  const width = Math.max(container.clientWidth || 520, 280);
  const rowHeight = 40;
  const gap = 8;
  const margin = { top: 8, right: 54, bottom: 8, left: 118 };
  const height = margin.top + margin.bottom + items.length * rowHeight;
  const plotW = width - margin.left - margin.right;

  const maxValue = Math.max(0.08, ...items.map((d) => Math.max(d.complaint_rate, d.market_median)));
  const xAt = (v) => (v / maxValue) * plotW;

  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height });
  const root = el("g", { transform: `translate(${margin.left},${margin.top})` });
  svg.appendChild(root);

  items.forEach((item, i) => {
    const y = i * rowHeight;
    const barH = rowHeight - gap;
    const isSelected = item.aspect === selected;

    const label = el("text", {
      class: "value-text",
      x: -12,
      y: y + barH / 2 + 4,
      "text-anchor": "end",
      "font-weight": isSelected ? 650 : 400,
    });
    label.textContent = ASPECT_NAMES[item.aspect] || item.aspect;
    root.appendChild(label);

    root.appendChild(el("path", {
      class: "bar-track",
      d: rightRoundedBar(0, y, plotW, barH, 4),
    }));

    const barWidth = Math.max(2, xAt(item.complaint_rate));
    root.appendChild(el("path", {
      class: "bar",
      d: rightRoundedBar(0, y, barWidth, barH, 4),
      opacity: isSelected || !selected ? 1 : 0.72,
    }));

    // Benchmark tick, inset 2px top and bottom so it reads as a separate mark.
    const refX = xAt(item.market_median);
    root.appendChild(el("line", {
      class: "reference-tick",
      x1: refX, y1: y + 2, x2: refX, y2: y + barH - 2,
    }));

    const value = el("text", {
      class: "value-text",
      x: plotW + 10,
      y: y + barH / 2 + 4,
      "font-weight": isSelected ? 650 : 400,
    });
    value.textContent = pct(item.complaint_rate);
    root.appendChild(value);

    const hit = el("rect", {
      x: -margin.left, y, width: width, height: rowHeight, fill: "transparent", cursor: "pointer",
    });
    hit.addEventListener("mousemove", (event) => {
      const direction = item.trend > 0.005 ? "rising" : item.trend < -0.005 ? "falling" : "flat";
      showTooltip(event, ASPECT_NAMES[item.aspect] || item.aspect, [
        `You: <strong>${pct1(item.complaint_rate)}</strong> of reviews`,
        `Market median: ${pct1(item.market_median)}`,
        `Trend: ${direction}`,
        `Click to see the monthly trend`,
      ]);
    });
    hit.addEventListener("mouseleave", hideTooltip);
    hit.addEventListener("click", () => onSelect(item.aspect));
    root.appendChild(hit);
  });

  container.appendChild(svg);
}

// ------------------------------------------------------------- table views

function renderTable(container, columns, rows) {
  const head = `<tr>${columns.map((c) => `<th>${c.header}</th>`).join("")}</tr>`;
  const body = rows
    .map((row) => `<tr>${columns.map((c) => `<td>${c.value(row)}</td>`).join("")}</tr>`)
    .join("");
  container.innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

// ---------------------------------------------------------------- rendering

function renderVerdict() {
  const { report } = state;
  const band = report.risk_band;
  const wording = { low: "Low risk", medium: "Watch closely", high: "High risk" }[band] || band;
  const pill = $("risk-pill");
  pill.dataset.band = band;
  $("risk-label").textContent = wording;
  $("verdict-period").textContent = `Analysis for ${monthLabel(report.period)}`;

  const top = report.findings[0];
  const name = state.aspects?.name || report.business_id;
  $("verdict-text").innerHTML = top
    ? `<strong>${name}</strong> has a <strong>${pct(report.decline_risk)}</strong> chance its rating ` +
      `falls over the next 3 months. The biggest driver is <strong>${top.label}</strong> &mdash; ` +
      `${pct(top.complaint_rate)} of recent reviews mention it, against a market median of ` +
      `${pct(top.market_median)}.`
    : `<strong>${name}</strong> has a <strong>${pct(report.decline_risk)}</strong> chance its rating ` +
      `falls over the next 3 months. No single issue stands out above the market benchmark.`;
}

function renderTiles() {
  const months = state.history.months;
  const recent = months.slice(-6);
  const prior = months.slice(-12, -6);

  const weighted = (rows) => {
    const scored = rows.filter((m) => m.mean_stars != null);
    const n = scored.reduce((sum, m) => sum + m.n_reviews, 0);
    if (!n) return null;
    return scored.reduce((sum, m) => sum + m.mean_stars * m.n_reviews, 0) / n;
  };

  $("tile-risk").textContent = pct(state.report.decline_risk);
  $("tile-risk-note").textContent = `${state.report.risk_band} risk over the next 3 months`;

  const now = weighted(recent);
  const before = weighted(prior);
  $("tile-stars").textContent = stars(now);
  if (now != null && before != null) {
    const delta = now - before;
    const cls = delta >= 0 ? "delta-up" : "delta-down";
    const arrow = delta >= 0 ? "▲" : "▼";
    $("tile-stars-note").innerHTML =
      `<span class="${cls}">${arrow} ${Math.abs(delta).toFixed(2)}</span> vs previous 6 months`;
  } else {
    $("tile-stars-note").textContent = "last 6 months";
  }

  const reviews = recent.reduce((sum, m) => sum + m.n_reviews, 0);
  $("tile-reviews").textContent = reviews;

  const recentKeys = new Set(months.slice(-3).map((m) => m.period));
  const live = state.alerts.filter((a) => recentKeys.has(a.period));
  $("tile-alerts").textContent = live.length;
  $("tile-alerts-note").textContent = live.length
    ? "issues spiking in the last 3 months"
    : "nothing spiking right now";
}

function renderRatingChart() {
  const points = state.history.months.map((m) => ({
    key: m.period, value: m.mean_stars, n_reviews: m.n_reviews,
  }));
  lineChart($("rating-chart"), {
    points,
    yDomain: [1, 5],
    format: (v) => v.toFixed(1),
    yLabel: "Average rating",
  });
  renderTable(
    $("rating-chart-table"),
    [
      { header: "Month", value: (r) => monthLabel(r.period) },
      { header: "Average rating", value: (r) => stars(r.mean_stars) },
      { header: "Reviews", value: (r) => r.n_reviews },
    ],
    state.history.months
  );
}

function renderAspectChart() {
  const items = [...state.aspects.aspects].sort((a, b) => b.complaint_rate - a.complaint_rate);
  barChart($("aspect-chart"), {
    items,
    selected: state.selectedAspect,
    onSelect: (aspect) => {
      state.selectedAspect = aspect;
      renderAspectChart();
      renderTrendChart();
    },
  });
  renderTable(
    $("aspect-chart-table"),
    [
      { header: "Issue", value: (r) => ASPECT_NAMES[r.aspect] || r.aspect },
      { header: "Your complaint rate", value: (r) => pct1(r.complaint_rate) },
      { header: "Market median", value: (r) => pct1(r.market_median) },
      { header: "Recent trend", value: (r) => (r.trend > 0 ? "+" : "") + pct1(r.trend) },
    ],
    items
  );
}

function renderTrendChart() {
  const aspect = state.selectedAspect;
  const name = ASPECT_NAMES[aspect] || aspect;
  $("trend-title").textContent = `${name} complaints over time`;
  $("trend-sub").textContent =
    "Share of each month's reviews that complain about this. Click another issue above to switch.";

  const points = state.history.months.map((m) => ({
    key: m.period, value: m[aspect], n_reviews: m.n_reviews,
  }));
  const peak = Math.max(0.1, ...points.map((p) => p.value ?? 0));
  lineChart($("trend-chart"), {
    points,
    yDomain: [0, niceCeiling(peak)],
    format: (v) => `${Math.round(v * 100)}%`,
    area: true,
    yLabel: `${name} complaints`,
  });
  renderTable(
    $("trend-chart-table"),
    [
      { header: "Month", value: (r) => monthLabel(r.period) },
      { header: `${name} complaint rate`, value: (r) => (r[aspect] == null ? "–" : pct1(r[aspect])) },
      { header: "Reviews", value: (r) => r.n_reviews },
    ],
    state.history.months
  );
}

function renderActions() {
  const list = $("actions");
  const findings = state.report.findings;
  if (!findings.length) {
    list.innerHTML = `<p class="empty">Nothing stands out above the market benchmark this month.</p>`;
    return;
  }
  list.innerHTML = findings
    .map((finding, i) => {
      const quotes = finding.evidence.length
        ? `<ul class="quotes">${finding.evidence.map((q) => `<li class="quote">&ldquo;${q}&rdquo;</li>`).join("")}</ul>`
        : `<p class="empty">No recent reviews quoted for this issue.</p>`;
      const gap = finding.complaint_rate - finding.market_median;
      const versus = gap >= 0
        ? `${pct1(gap)} above the market median`
        : `${pct1(-gap)} below the market median`;
      // Only quote a star cost when there is one worth quoting; "costing about
      // 0.00 stars" is noise that makes the whole card look broken.
      const cost = finding.estimated_star_cost >= 0.01
        ? ` &middot; costing about ${finding.estimated_star_cost.toFixed(2)} stars`
        : "";
      return `
        <li class="action">
          <div class="action-rank">${i + 1}</div>
          <div>
            <h3 class="action-title">${ASPECT_NAMES[finding.aspect] || finding.label}</h3>
            <p class="action-stats">
              ${pct1(finding.complaint_rate)} of recent reviews &middot; ${versus}${cost}
            </p>
            <p class="action-do"><strong>Do this:</strong> ${finding.suggested_action}</p>
            ${quotes}
          </div>
        </li>`;
    })
    .join("");
}

function renderAgentPlan() {
  const container = $("agent-plan");
  const badge = $("planner-badge");
  const plan = state.plan;

  if (!plan) {
    container.innerHTML = `<p class="empty">The agent could not produce a plan for this restaurant.</p>`;
    badge.hidden = true;
    return;
  }

  // Name the planner that actually ran: a deterministic fallback plan must
  // never be mistaken for one the language model wrote.
  badge.hidden = false;
  badge.textContent = plan.planner === "claude" ? "Claude agent" : "rule-based";

  if (!plan.plan.length) {
    container.innerHTML =
      `<p class="plan-summary">${plan.summary || "No verified recommendations for this month."}</p>` +
      `<p class="empty">No plan item passed verification, so nothing is shown here.</p>`;
    return;
  }

  const items = plan.plan
    .map((item) => {
      const quotes = (item.quotes || []).length
        ? `<ul class="quotes">${item.quotes.map((q) => `<li class="quote">&ldquo;${q}&rdquo;</li>`).join("")}</ul>`
        : "";
      const cites = (item.evidence_review_ids || []).length
        ? `<p class="plan-cites">Backed by ${item.evidence_review_ids.length} cited review${
            item.evidence_review_ids.length === 1 ? "" : "s"
          }: ${item.evidence_review_ids.join(", ")}</p>`
        : "";
      return `
        <li class="action">
          <div class="action-rank">${item.rank}</div>
          <div>
            <h3 class="action-title">${ASPECT_NAMES[item.aspect] || item.aspect}
              <span class="verified-chip"><span class="pill-dot" aria-hidden="true"></span>verified</span>
            </h3>
            <p class="plan-claim">${item.claim}</p>
            <p class="action-do"><strong>Do this:</strong> ${item.action}</p>
            ${quotes}
            ${cites}
          </div>
        </li>`;
    })
    .join("");

  const rejected = plan.rejected.length
    ? `<p class="plan-meta"><span class="verified-chip is-rejected">
         <span class="pill-dot" aria-hidden="true"></span>${plan.rejected.length} claim(s) rejected by the verifier
       </span></p>`
    : "";

  container.innerHTML =
    `<p class="plan-summary">${plan.summary}</p>` +
    `<ol class="actions">${items}</ol>` +
    rejected +
    `<p class="plan-meta">
       <span>Groundedness ${plan.groundedness === null ? "n/a" : Math.round(plan.groundedness * 100) + "%"}</span>
       <span>${plan.n_accepted} of ${plan.n_drafted} claims verified</span>
       <span>${plan.tools_called} tool calls</span>
       <span>${plan.latency_seconds}s</span>
     </p>`;
}

async function loadAgentPlan(businessId) {
  const container = $("agent-plan");
  container.innerHTML = `<p class="empty">Running the agent&hellip;</p>`;
  try {
    state.plan = await getJSON(`/businesses/${businessId}/plan`);
  } catch (error) {
    state.plan = null;
    container.innerHTML =
      `<p class="empty">The agent is unavailable: ${error.message}</p>`;
    return;
  }
  renderAgentPlan();
}

function renderAlerts() {
  const container = $("alerts");
  const rows = state.alerts.slice(0, 8);
  if (!rows.length) {
    container.innerHTML = `<p class="empty">No complaint spikes detected for this restaurant.</p>`;
    return;
  }
  const body = rows
    .map((alert) => `
      <tr>
        <td>${monthLabel(alert.period)}</td>
        <td>${ASPECT_NAMES[alert.aspect] || alert.aspect}</td>
        <td>${pct1(alert.rate)}</td>
        <td>${pct1(alert.baseline)}</td>
        <td><span class="pill" data-band="${alert.severity}"><span class="pill-dot"></span>${alert.severity}</span></td>
      </tr>`)
    .join("");
  container.innerHTML = `
    <div class="table-view">
      <table>
        <thead><tr><th>Month</th><th>Issue</th><th>Complaint rate</th><th>Your normal</th><th>Severity</th></tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>`;
}

// ------------------------------------------------------------------- wiring

async function loadVenue(businessId) {
  state.businessId = businessId;
  const [report, history, aspects, alerts] = await Promise.all([
    getJSON(`/businesses/${businessId}/report`),
    getJSON(`/businesses/${businessId}/history?months=24`),
    getJSON(`/businesses/${businessId}/aspects`),
    getJSON(`/alerts?limit=1000`),
  ]);

  state.report = report;
  state.history = history;
  state.aspects = aspects;
  state.alerts = alerts
    .filter((a) => a.business_id === businessId)
    .sort((a, b) => (a.period < b.period ? 1 : -1));
  // Default the trend chart to the issue the model says to fix first.
  state.selectedAspect = report.findings[0]?.aspect || aspects.aspects[0].aspect;

  renderVerdict();
  renderTiles();
  renderRatingChart();
  renderAspectChart();
  renderTrendChart();
  renderActions();
  renderAlerts();

  // The agent runs several model calls, so it loads after the rest of the page
  // rather than holding the whole dashboard up behind it.
  loadAgentPlan(businessId);
}

function wireTableToggles() {
  document.querySelectorAll("[data-table-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const target = $(`${button.dataset.tableToggle}-table`);
      const showing = !target.hidden;
      target.hidden = showing;
      button.textContent = showing ? "Table" : "Chart";
      $(button.dataset.tableToggle).hidden = !showing;
    });
  });
}

function wirePlanRefresh() {
  $("plan-refresh").addEventListener("click", () => {
    if (state.businessId) loadAgentPlan(state.businessId);
  });
}

function wireTheme() {
  const button = $("theme-toggle");
  const stored = (() => {
    try { return localStorage.getItem("theme"); } catch { return null; }
  })();
  if (stored) document.documentElement.dataset.theme = stored;

  button.addEventListener("click", () => {
    const isDark = document.documentElement.dataset.theme === "dark" ||
      (!document.documentElement.dataset.theme &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
    const next = isDark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* private mode */ }
    if (state.history) { renderRatingChart(); renderAspectChart(); renderTrendChart(); }
  });
}

function wireResize() {
  let timer = null;
  window.addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (!state.history) return;
      renderRatingChart();
      renderAspectChart();
      renderTrendChart();
    }, 150);
  });
}

async function init() {
  wireTheme();
  wireTableToggles();
  wirePlanRefresh();
  wireResize();

  try {
    state.venues = await getJSON("/businesses?limit=200");
  } catch (error) {
    $("verdict-text").textContent =
      "Could not reach the API. Train the models first: python -m reputation.pipeline.train";
    return;
  }

  const select = $("venue-select");
  select.innerHTML = state.venues
    .map((v) => `<option value="${v.business_id}">${v.name} — ${pct(v.decline_risk)} risk</option>`)
    .join("");
  select.addEventListener("change", () => loadVenue(select.value));

  if (state.venues.length) await loadVenue(state.venues[0].business_id);
}

init();
