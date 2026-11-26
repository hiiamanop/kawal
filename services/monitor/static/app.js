const state = { cases: [], view: "feed" };

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function statusLabel(status) {
  return (
    {
      NEW: "Baru",
      MONITORING: "Dipantau",
      READY_TO_FORWARD: "Siap Diteruskan",
      FORWARDED: "Sudah Diteruskan",
      DISMISSED: "Diabaikan",
    }[status] || status
  );
}

function actionsForStatus(status) {
  if (status === "FORWARDED" || status === "DISMISSED") {
    return '<p class="meta">Status terminal. Kasus ini bersifat read-only.</p>';
  }
  if (status === "READY_TO_FORWARD") {
    return `<p>
      <button class="primary" data-action="forward">Buat Draft</button>
      <button class="primary" data-action="FORWARDED">Mark forwarded</button>
      <button class="ghost" data-action="MONITORING">Kembali monitor</button>
      <button class="ghost" data-action="DISMISSED">Abaikan</button>
    </p>`;
  }
  if (status === "MONITORING") {
    return `<p>
      <button class="primary" data-action="READY_TO_FORWARD">Siap Diteruskan</button>
      <button class="ghost" data-action="DISMISSED">Abaikan</button>
    </p>`;
  }
  return `<p>
    <button class="ghost" data-action="MONITORING">Monitor</button>
    <button class="primary" data-action="READY_TO_FORWARD">Siap Diteruskan</button>
    <button class="ghost" data-action="DISMISSED">Abaikan</button>
  </p>`;
}

function casesUrl() {
  const params = new URLSearchParams();
  const viewStatus = {
    monitoring: "MONITORING",
    ready: "READY_TO_FORWARD",
    forwarded: "FORWARDED",
    archive: "DISMISSED",
  }[state.view];
  if (viewStatus) params.set("status", viewStatus);
  if (state.view === "errors") params.set("analysis_status", "FAILED");
  return `/api/monitor/cases${params.size ? `?${params}` : ""}`;
}

async function load() {
  const [statsRes, casesRes] = await Promise.all([
    fetch("/api/monitor/stats"),
    fetch(casesUrl()),
  ]);
  if (!statsRes.ok || !casesRes.ok) throw new Error("Gagal memuat data monitor");
  const stats = await statsRes.json();
  const cases = await casesRes.json();
  state.cases = cases.cases || [];
  renderStats(stats);
  renderFeed();
}

function renderStats(stats) {
  const items = [
    ["Total", stats.total],
    ["Complaint", stats.complaints],
    ["Needs review", stats.needs_review],
    ["Siap diteruskan", stats.ready_to_forward],
  ];
  $("stats").innerHTML = items
    .map(
      ([label, value]) =>
        `<div class="stat"><span>${escapeHtml(label)}</span><b>${Number(value || 0)}</b></div>`,
    )
    .join("");
}

function filteredCases() {
  const status = $("status-filter").value;
  const category = $("category-filter").value;
  const risk = $("risk-filter").value;
  return state.cases.filter((item) => {
    const c = item.case;
    const triage = c ? c.triage : null;
    if (status && item.operator_status !== status) return false;
    if (category && triage?.category !== category) return false;
    if (risk && triage?.risk !== risk) return false;
    return true;
  });
}

function renderFeed() {
  const feed = $("feed");
  const cases = filteredCases();
  if (!cases.length) {
    feed.innerHTML =
      '<div class="empty">Belum ada data yang cocok. Klik <b>Scan sekarang</b> untuk mulai.</div>';
    return;
  }
  feed.innerHTML = cases
    .map((item) => {
      const c = item.case;
      const triage = c ? c.triage : null;
      const score = item.viral_score == null ? "-" : item.viral_score;
      const category = triage ? triage.category : "ANALYSIS ERROR";
      const risk = triage ? triage.risk : "-";
      const trust = c ? c.status : item.analysis_status;
      return `
        <article class="card" role="button" tabindex="0" data-case="${escapeHtml(item.case_id)}">
          <div class="card-head">
            <div class="badges">
              <span class="badge">${escapeHtml(category)}</span>
              <span class="badge risk-${escapeHtml(risk)}">${escapeHtml(risk)}</span>
              <span class="badge trust-${escapeHtml(trust)}">${escapeHtml(trust)}</span>
            </div>
            <span class="score">${escapeHtml(score)}</span>
          </div>
          <div class="meta">@${escapeHtml(item.post.author_handle || "-")} · ${escapeHtml(item.post.posted_at || "")} · ${escapeHtml(item.department || "Belum ada rekomendasi dinas")}</div>
          <div class="excerpt">${escapeHtml((item.post.text || "").slice(0, 360))}</div>
          <div class="meta">${escapeHtml(statusLabel(item.operator_status))}</div>
        </article>`;
    })
    .join("");
  feed.querySelectorAll(".card").forEach((card) => {
    const open = () => openDetail(card.dataset.case);
    card.addEventListener("click", open);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });
  });
}

async function scan(query) {
  if (!query || !query.trim()) return;
  const sourceMode = $("source-mode").value;
  const fixtureDir = $("fixture-dir").value.trim();
  if (sourceMode === "replay" && !fixtureDir) {
    alert("Folder fixture wajib diisi untuk mode replay.");
    return;
  }
  const button = $("scan-button");
  button.disabled = true;
  button.textContent = "Memindai...";
  try {
    const response = await fetch("/api/monitor/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query.trim(),
        limit: 10,
        source_mode: sourceMode,
        analysis_mode: $("analysis-mode").value,
        fixture_dir: sourceMode === "replay" ? fixtureDir : null,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail?.message || data.detail || "Scan gagal");
    }
    await load();
  } catch (error) {
    alert(`Scan gagal: ${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "Scan sekarang";
  }
}

function renderDetail(item, actions, queries) {
  const c = item.case;
  const triage = c ? c.triage : null;
  const extraction = c ? c.extraction : null;
  const verification = c ? c.verification : null;
  const arbitration = c ? c.arbitration : null;
  const cache = item.run.cache;
  const trace = item.run.trace || [];
  const engagement = item.post.raw || {};
  const evidence = extraction?.evidence_spans || [];
  const reasons = c?.reason_codes || [];
  return `
    <h2>${escapeHtml(item.post.author_handle || "Keluhan")}</h2>
    <div class="badges">
      <span class="badge">${escapeHtml(triage ? triage.category : "ANALYSIS ERROR")}</span>
      <span class="badge risk-${escapeHtml(triage ? triage.risk : "-")}">${escapeHtml(triage ? triage.risk : "-")}</span>
      <span class="badge trust-${escapeHtml(c ? c.status : item.analysis_status)}">${escapeHtml(c ? c.status : item.analysis_status)}</span>
      <span class="badge">${escapeHtml(statusLabel(item.operator_status))}</span>
    </div>
    <p class="excerpt">${escapeHtml(item.post.text)}</p>
    <p><a href="${escapeHtml(item.post.url)}" target="_blank" rel="noreferrer">Buka post asli</a></p>
    <div class="key">Skor viral / engagement</div>
    <p>${escapeHtml(item.viral_score ?? "-")} · likes ${escapeHtml(engagement.likes ?? "-")} · views ${escapeHtml(engagement.views ?? "-")}</p>
    <div class="key">Rekomendasi dinas</div>
    <p>${escapeHtml(item.department || "-")}</p>
    <div class="key">Lokasi / kualitas lokasi</div>
    <p>${escapeHtml(extraction?.location || "-")} · ${escapeHtml(c?.location_quality || "-")}</p>
    <div class="key">Trust decision</div>
    <p>${escapeHtml(c?.trust_decision || "-")}</p>
    <div class="key">Source intent</div>
    <p>${escapeHtml(c?.source_intent || "-")}</p>
    <div class="key">Evidence spans</div>
    <p>${escapeHtml(evidence.length ? evidence.join(", ") : "-")}</p>
    <div class="key">Reason codes</div>
    <p>${escapeHtml(reasons.length ? reasons.join(", ") : "-")}</p>
    <div class="key">Triage</div>
    <pre>${escapeHtml(JSON.stringify(triage, null, 2))}</pre>
    <div class="key">Verifikasi</div>
    <pre>${escapeHtml(JSON.stringify(verification, null, 2))}</pre>
    <div class="key">Arbiter</div>
    <pre>${escapeHtml(arbitration ? JSON.stringify(arbitration, null, 2) : "-")}</pre>
    <div class="key">Cache</div>
    <pre>${escapeHtml(cache ? JSON.stringify(cache, null, 2) : "not attempted")}</pre>
    <div class="key">Operator note</div>
    <p>${escapeHtml(item.notes || "-")}</p>
    <div class="key">Trace</div>
    <pre>${escapeHtml(trace.map((event) => event.event).join(" → "))}</pre>
    <div class="key">Actions</div>
    <pre>${escapeHtml(JSON.stringify(actions, null, 2))}</pre>
    <div class="key">Queries</div>
    <pre>${escapeHtml(JSON.stringify(queries, null, 2))}</pre>
    <div class="key">Aksi</div>
    ${actionsForStatus(item.operator_status)}
  `;
}

async function openDetail(caseId) {
  const response = await fetch(`/api/monitor/cases/${caseId}`);
  if (!response.ok) return;
  const data = await response.json();
  $("detail-content").innerHTML = renderDetail(data.case, data.actions, data.queries);
  $("detail-panel").hidden = false;
  $("detail-content").querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      const action = button.dataset.action;
      if (action === "forward") {
        const note = prompt(
          "Catatan operator untuk draft penerusan:",
          "Mohon diteruskan ke dinas terkait.",
        );
        if (note === null) return;
        const response = await fetch(`/api/monitor/cases/${caseId}/forward`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ note }),
        });
        const data = await response.json();
        if (!response.ok) {
          alert(data.detail?.message || "Gagal membuat draft");
          return;
        }
        const filename = data.markdown_path.split("/").pop();
        window.location.href = `/api/monitor/forwards/${filename}`;
        await openDetail(caseId);
        await load();
        return;
      }
      const response = await fetch(`/api/monitor/cases/${caseId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: action }),
      });
      if (!response.ok) {
        const data = await response.json();
        alert(data.detail?.message || "Status tidak dapat diubah");
        return;
      }
      await openDetail(caseId);
      await load();
    });
  });
}

$("scan-button").addEventListener("click", () => {
  const query = prompt("Query pencarian X:", "jalan rusak bandung");
  if (query) scan(query);
});
$("composer-scan").addEventListener("click", () => scan($("query-input").value));
$("query-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") scan($("query-input").value);
});
$("refresh-button").addEventListener("click", () => load());
["category-filter", "risk-filter", "status-filter"].forEach((id) => {
  $(id).addEventListener("change", renderFeed);
});
$("source-mode").addEventListener("change", () => {
  $("fixture-dir").hidden = $("source-mode").value !== "replay";
});
$("close-detail").addEventListener("click", () => {
  $("detail-panel").hidden = true;
});
document.querySelectorAll(".nav").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.view = button.dataset.view;
    load().catch((error) => {
      $("feed").innerHTML = `<div class="empty error">${escapeHtml(error.message)}</div>`;
    });
  });
});

load().catch((error) => {
  $("feed").innerHTML = `<div class="empty error">${escapeHtml(error.message)}</div>`;
});
