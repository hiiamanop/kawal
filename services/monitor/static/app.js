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

async function load() {
  const [statsRes, casesRes] = await Promise.all([
    fetch("/api/monitor/stats"),
    fetch("/api/monitor/cases"),
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

function renderFeed() {
  const feed = $("feed");
  if (!state.cases.length) {
    feed.innerHTML =
      '<div class="empty">Belum ada data. Klik <b>Scan sekarang</b> untuk mulai.</div>';
    return;
  }
  feed.innerHTML = state.cases
    .map((item) => {
      const c = item.case;
      const triage = c ? c.triage : null;
      const score = item.viral_score == null ? "-" : item.viral_score;
      const category = triage ? triage.category : "-";
      const risk = triage ? triage.risk : "-";
      const trust = c ? c.status : item.analysis_status;
      return `
        <article class="card" data-case="${escapeHtml(item.case_id)}">
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
    card.addEventListener("click", () => openDetail(card.dataset.case));
  });
}

async function scan(query) {
  if (!query || !query.trim()) return;
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
        source_mode: "live",
        analysis_mode: "rule",
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

async function openDetail(caseId) {
  const response = await fetch(`/api/monitor/cases/${caseId}`);
  if (!response.ok) return;
  const data = await response.json();
  const item = data.case;
  const c = item.case;
  const triage = c ? c.triage : null;
  const extraction = c ? c.extraction : null;
  const verification = c ? c.verification : null;
  $("detail-content").innerHTML = `
    <h2>${escapeHtml(item.post.author_handle || "Keluhan")}</h2>
    <div class="badges">
      <span class="badge">${escapeHtml(triage ? triage.category : "-")}</span>
      <span class="badge risk-${escapeHtml(triage ? triage.risk : "-")}">${escapeHtml(triage ? triage.risk : "-")}</span>
      <span class="badge trust-${escapeHtml(c ? c.status : item.analysis_status)}">${escapeHtml(c ? c.status : item.analysis_status)}</span>
    </div>
    <p class="excerpt">${escapeHtml(item.post.text)}</p>
    <p><a href="${escapeHtml(item.post.url)}" target="_blank" rel="noreferrer">Buka post asli</a></p>
    <div class="key">Rekomendasi dinas</div>
    <p>${escapeHtml(item.department || "-")}</p>
    <div class="key">Lokasi</div>
    <p>${escapeHtml(extraction?.location || "-")}</p>
    <div class="key">Verifikasi</div>
    <p>${escapeHtml(verification ? verification.evidence_supported : "-")} ${escapeHtml((verification?.issues || []).join(", "))}</p>
    <div class="key">Trace</div>
    <pre>${escapeHtml((item.run.trace || []).map((event) => event.event).join(" → "))}</pre>
    <div class="key">Aksi</div>
    <p>
      <button class="ghost" data-action="MONITORING">Monitor</button>
      <button class="primary" data-action="READY_TO_FORWARD">Siap Diteruskan</button>
    </p>
  `;
  $("detail-panel").hidden = false;
  $("detail-content").querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      await fetch(`/api/monitor/cases/${caseId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: button.dataset.action }),
      });
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
$("close-detail").addEventListener("click", () => {
  $("detail-panel").hidden = true;
});
document.querySelectorAll(".nav").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.view = button.dataset.view;
  });
});

load().catch((error) => {
  $("feed").innerHTML = `<div class="empty error">${escapeHtml(error.message)}</div>`;
});
