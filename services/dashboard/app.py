from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import psycopg
from psycopg.rows import dict_row


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KAWAL Case Inspector & Telemetry</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .badge-TICKETED { background-color: #059669; color: white; }
        .badge-READY { background-color: #2563eb; color: white; }
        .badge-ANALYZING { background-color: #d97706; color: white; }
        .badge-WAITING_CLARIFICATION { background-color: #7c3aed; color: white; }
        .badge-REJECTED { background-color: #dc2626; color: white; }
        .badge-BLOCKED { background-color: #4b5563; color: white; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans">
    <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-10 px-6 py-4 flex items-center justify-between">
        <div class="flex items-center space-x-3">
            <span class="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></span>
            <h1 class="text-xl font-bold tracking-tight text-white">KAWAL <span class="text-slate-400 font-normal text-sm">| Live Case Inspector & Decision Lineage</span></h1>
        </div>
        <div class="flex items-center space-x-4 text-xs text-slate-400">
            <span>Redpanda: <b class="text-emerald-400">19092</b></span>
            <span>PostgreSQL: <b class="text-emerald-400">54322</b></span>
            <span>OPA: <b class="text-emerald-400">8181</b></span>
            <button onclick="fetchCases()" class="bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded transition">Refresh</button>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 py-6 space-y-6">
        <!-- Stats Row -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4" id="stats-container">
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div class="text-slate-400 text-xs font-medium uppercase">Total Kasus</div>
                <div class="text-2xl font-bold text-white mt-1" id="stat-total">-</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div class="text-slate-400 text-xs font-medium uppercase">Tiket Terbit (TICKETED)</div>
                <div class="text-2xl font-bold text-emerald-400 mt-1" id="stat-ticketed">-</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div class="text-slate-400 text-xs font-medium uppercase">Menunggu Klarifikasi</div>
                <div class="text-2xl font-bold text-purple-400 mt-1" id="stat-clarify">-</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div class="text-slate-400 text-xs font-medium uppercase">Ditolak / Non-Aduan</div>
                <div class="text-2xl font-bold text-rose-400 mt-1" id="stat-rejected">-</div>
            </div>
        </div>

        <!-- Cases Table & Detail Layout -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div class="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                <div class="px-5 py-3.5 border-b border-slate-800 flex justify-between items-center">
                    <h2 class="font-semibold text-sm text-slate-200">Daftar Kasus Masuk</h2>
                    <span class="text-xs text-slate-400">Klik baris untuk melihat lineage audit</span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="bg-slate-950 text-slate-400 border-b border-slate-800">
                            <tr>
                                <th class="px-4 py-2.5">ID Kasus</th>
                                <th class="px-3 py-2.5">Tenant / Pengirim</th>
                                <th class="px-3 py-2.5">Kategori</th>
                                <th class="px-3 py-2.5">Risiko</th>
                                <th class="px-3 py-2.5">Status</th>
                                <th class="px-4 py-2.5">Tiket ID</th>
                            </tr>
                        </thead>
                        <tbody id="cases-tbody" class="divide-y divide-slate-800/60 font-mono">
                            <tr><td colspan="6" class="px-4 py-8 text-center text-slate-500">Memuat data kasus...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Detail / Audit Trace Panel -->
            <div class="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-xl flex flex-col h-[650px] overflow-hidden">
                <div class="px-5 py-3.5 border-b border-slate-800 flex justify-between items-center bg-slate-900">
                    <h2 class="font-semibold text-sm text-slate-200">Decision Lineage & Audit Trail</h2>
                    <span id="detail-case-title" class="text-xs text-emerald-400 font-mono">-</span>
                </div>
                <div class="p-5 flex-1 overflow-y-auto space-y-4 text-xs" id="detail-content">
                    <div class="text-slate-500 text-center py-20">Pilih salah satu kasus pada tabel untuk memeriksa bukti snapshot, analisis ML, dan jejak eksekusi tiket.</div>
                </div>
            </div>
        </div>
    </main>

    <script>
        async function fetchCases() {
            try {
                const res = await fetch('/api/cases?limit=50');
                const data = await res.json();
                renderCases(data.cases);
                renderStats(data.stats);
            } catch (err) {
                console.error(err);
            }
        }

        function renderStats(s) {
            if (!s) return;
            document.getElementById('stat-total').innerText = s.total || 0;
            document.getElementById('stat-ticketed').innerText = s.ticketed || 0;
            document.getElementById('stat-clarify').innerText = s.waiting_clarification || 0;
            document.getElementById('stat-rejected').innerText = s.rejected || 0;
        }

        function renderCases(cases) {
            const tbody = document.getElementById('cases-tbody');
            if (!cases || cases.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="px-4 py-8 text-center text-slate-500">Belum ada kasus yang tercatat di basis data.</td></tr>';
                return;
            }
            tbody.innerHTML = cases.map(c => `
                <tr onclick="loadCaseDetail('${c.case_id}')" class="hover:bg-slate-800/50 cursor-pointer transition">
                    <td class="px-4 py-3 font-medium text-slate-300">${c.case_id.substring(0, 18)}...</td>
                    <td class="px-3 py-3 text-slate-400">${c.tenant_id}<br><span class="text-[10px] text-slate-500">${c.conversation_id || '-'}</span></td>
                    <td class="px-3 py-3 text-slate-300">${c.category || '-'}</td>
                    <td class="px-3 py-3">${c.risk ? `<span class="text-[10px] px-1.5 py-0.5 rounded ${c.risk === 'HIGH' || c.risk === 'URGENT' ? 'bg-rose-950 text-rose-300 border border-rose-800' : 'bg-slate-800 text-slate-300'}">${c.risk}</span>` : '-'}</td>
                    <td class="px-3 py-3"><span class="px-2 py-0.5 rounded text-[10px] font-semibold badge-${c.processing_state}">${c.processing_state}</span></td>
                    <td class="px-4 py-3 text-emerald-400 font-mono text-[11px]">${c.ticket_id ? c.ticket_id.substring(0, 8) + '...' : '-'}</td>
                </tr>
            `).join('');
        }

        async function loadCaseDetail(caseId) {
            document.getElementById('detail-case-title').innerText = caseId;
            const detailContainer = document.getElementById('detail-content');
            detailContainer.innerHTML = '<div class="text-slate-400 py-10 text-center">Memuat riwayat kasus...</div>';

            try {
                const res = await fetch(`/api/cases/${caseId}`);
                const d = await res.json();

                let html = `
                    <div class="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2">
                        <div class="font-semibold text-slate-300">Ringkasan Kasus</div>
                        <div class="grid grid-cols-2 gap-2 text-[11px] text-slate-400">
                            <div>Kategori: <b class="text-slate-200">${d.category || '-'}</b></div>
                            <div>Risiko: <b class="text-slate-200">${d.risk || '-'}</b></div>
                            <div>Revisi Kasus: <b class="text-slate-200">${d.revision || 1}</b></div>
                            <div>Tiket ID: <b class="text-emerald-400">${d.ticket_id || '-'}</b></div>
                        </div>
                    </div>
                `;

                if (d.snapshot) {
                    html += `
                        <div class="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2">
                            <div class="font-semibold text-slate-300">Pesan Aduan Warga (Evidence Hash: <span class="font-mono text-[10px] text-slate-500">${d.snapshot.evidence_hash.substring(0, 12)}...</span>)</div>
                            <div class="space-y-1.5">
                                ${d.snapshot.messages.map((m, idx) => `
                                    <div class="bg-slate-900 p-2.5 rounded border border-slate-800/80">
                                        <div class="text-[10px] text-slate-500 mb-1">Turn #${idx+1} • ${m.received_at}</div>
                                        <div class="text-slate-200 text-xs">${m.text}</div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }

                if (d.audit_traces && d.audit_traces.length > 0) {
                    html += `
                        <div class="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2">
                            <div class="font-semibold text-slate-300">Audit Trail Events</div>
                            <div class="space-y-2">
                                ${d.audit_traces.map(a => `
                                    <div class="bg-slate-900 p-2.5 rounded border border-slate-800/80">
                                        <div class="flex justify-between items-center mb-1">
                                            <span class="font-semibold text-emerald-400 text-[11px]">${a.event_type}</span>
                                            <span class="text-[10px] text-slate-500">${a.created_at}</span>
                                        </div>
                                        <pre class="bg-slate-950 p-2 rounded text-[10px] text-slate-400 overflow-x-auto">${JSON.stringify(a.payload, null, 2)}</pre>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }

                detailContainer.innerHTML = html;
            } catch (err) {
                detailContainer.innerHTML = '<div class="text-rose-400 py-10 text-center">Gagal memuat detail kasus.</div>';
            }
        }

        fetchCases();
        setInterval(fetchCases, 10000);
    </script>
</body>
</html>
"""


def create_dashboard_app(database_url: str) -> FastAPI:
    app = FastAPI(title="KAWAL Web Inspector", version="1.0.0")

    def get_db():
        return psycopg.connect(database_url, row_factory=dict_row)

    @app.get("/", response_class=HTMLResponse)
    def index_page() -> str:
        return HTML_TEMPLATE

    @app.get("/api/cases")
    def list_cases(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT case_id, tenant_id, conversation_id, revision, processing_state,
                           category, risk, sensitivity, ticket_id, created_at, updated_at
                    FROM cases
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                cases = cur.fetchall()

                cur.execute("SELECT count(*) as total FROM cases")
                total = cur.fetchone()["total"]

                cur.execute("SELECT count(*) as ticketed FROM cases WHERE processing_state = 'TICKETED'")
                ticketed = cur.fetchone()["ticketed"]

                cur.execute("SELECT count(*) as waiting_clarification FROM cases WHERE processing_state = 'WAITING_CLARIFICATION'")
                waiting_clarification = cur.fetchone()["waiting_clarification"]

                cur.execute("SELECT count(*) as rejected FROM cases WHERE processing_state = 'REJECTED'")
                rejected = cur.fetchone()["rejected"]

                stats = {
                    "total": total,
                    "ticketed": ticketed,
                    "waiting_clarification": waiting_clarification,
                    "rejected": rejected,
                }

                return {"cases": cases, "stats": stats}

    @app.get("/api/cases/{case_id}")
    def get_case_detail(case_id: str) -> dict[str, Any]:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT case_id, tenant_id, conversation_id, revision, processing_state,
                           category, risk, sensitivity, ticket_id, created_at, updated_at
                    FROM cases
                    WHERE case_id = %s
                    """,
                    (case_id,),
                )
                case_row = cur.fetchone()
                if case_row is None:
                    raise HTTPException(status_code=404, detail="Case not found")

                cur.execute(
                    """
                    SELECT revision, state, evidence_hash, messages_payload, created_at
                    FROM case_snapshots
                    WHERE case_id = %s
                    ORDER BY revision DESC
                    LIMIT 1
                    """,
                    (case_id,),
                )
                snap_row = cur.fetchone()
                snapshot = None
                if snap_row:
                    snapshot = {
                        "revision": snap_row["revision"],
                        "state": snap_row["state"],
                        "evidence_hash": snap_row["evidence_hash"],
                        "messages": snap_row["messages_payload"],
                        "created_at": snap_row["created_at"].isoformat() if hasattr(snap_row["created_at"], "isoformat") else str(snap_row["created_at"]),
                    }

                cur.execute(
                    """
                    SELECT event_type, payload, created_at
                    FROM audit_traces
                    WHERE case_id = %s
                    ORDER BY created_at ASC
                    """,
                    (case_id,),
                )
                traces_rows = cur.fetchall()
                audit_traces = [
                    {
                        "event_type": r["event_type"],
                        "payload": r["payload"],
                        "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"]),
                    }
                    for r in traces_rows
                ]

                result = dict(case_row)
                result["snapshot"] = snapshot
                result["audit_traces"] = audit_traces
                return result

    return app
