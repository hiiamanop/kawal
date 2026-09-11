# Milestone KAWAL

Dokumen ini adalah catatan progres resmi KAWAL. Item hanya berstatus **SELESAI** bila ada artefak dan bukti yang dapat diverifikasi di repositori.

- **Status proyek:** Spesifikasi dan perencanaan
- **Terakhir diperbarui:** 2026-09-11

## Status

| Status | Arti |
|---|---|
| `SELESAI` | Keluaran tersedia dan bukti tercantum. |
| `DALAM_PROGRES` | Pekerjaan sedang aktif, tetapi belum memenuhi kriteria penerimaan. |
| `BELUM_DIMULAI` | Belum terdapat implementasi atau artefak yang memadai. |
| `DITUNDA` | Ditahan oleh keputusan, dependensi, atau prioritas. |

## Format Pencatatan

| Bidang | Arti |
|---|---|
| ID | Identitas paket kerja atau kebutuhan PRD. |
| Pekerjaan | Keluaran yang harus tersedia. |
| Status | Status berdasarkan definisi di atas. |
| Bukti | File atau hasil validasi yang dapat diperiksa. |
| Catatan | Dependensi, risiko, atau langkah lanjut material. |

## M0 — Spesifikasi dan perencanaan

| ID | Pekerjaan | Status | Bukti | Catatan |
|---|---|---|---|---|
| DOC-PRD | PRD KAWAL | `SELESAI` | [`docs/PRD.md`](docs/PRD.md) | Menetapkan tujuan riset, arsitektur, kebutuhan, NFR, dan evaluasi. |
| DOC-SEC | Kebijakan keamanan | `SELESAI` | [`docs/SECURITY.md`](docs/SECURITY.md) | Memuat kontrol keamanan, privasi, dan keselamatan AI. |
| DOC-AGT | Protokol kolaborasi agen | `SELESAI` | [`AGENTS.md`](AGENTS.md) | Mengatur delegasi, progres, validasi, dan pelaporan. |
| DOC-PLAN | Rencana implementasi P0 | `SELESAI` | [`plan.md`](plan.md) | Roadmap M0–M6 serta kriteria penerimaan. |
| DOC-DSN | Desain arsitektur KAWAL | `SELESAI` | [DESIGN.md](DESIGN.md) | Menetapkan desain produk, batas UX, dan inspeksi/trace untuk KAWAL. |
| DEC-OPEN | Keputusan desain terbuka PRD | `DALAM_PROGRES` | [PRD §23](PRD.md#23-multimodal--vlm-architecture), [plan.md](plan.md) | Egress gambar telah dibekukan: tanpa visi lokal; gate deterministik non-visi menahan gambar sensitif/ambigu dan beralih ke klarifikasi teks. ModelGateway, idempotensi WhatsApp, dan artifact pinning tersisa. |

## M1 — Fondasi data dan vertical slice

| ID | Pekerjaan | Status | Bukti | Catatan |
|---|---|---|---|---|
| FND-01 | Infrastruktur lokal, migrasi, dan kontrak bertipe | `DALAM_PROGRES` | [`contracts/models.py`](../contracts/models.py), [`docker-compose.yml`](../docker-compose.yml), [`infra/migrations/001_m1_core_schema.sql`](../infra/migrations/001_m1_core_schema.sql) | Kontrak, Compose, dan migrasi PostgreSQL awal tersedia; layanan belum dijalankan dan koneksi aplikasi belum dibuat. |
| FND-02 | Transaksi state, inbox, dan outbox | `DALAM_PROGRES` | [`infra/migrations/001_m1_core_schema.sql`](../infra/migrations/001_m1_core_schema.sql) | Tabel dan constraint inbox/outbox tersedia; transaction runner dan relay belum dibuat. |
| FND-03 | Registry model/policy dan manifest | `DALAM_PROGRES` | [`policies/.manifest`](../policies/.manifest), [`policies/policy.rego`](../policies/policy.rego) | Bundle policy M1 berversi tersedia; registry model/manifest eksperimen belum dibuat. |
| TKT-01 | Ticket Simulator minimal | `DALAM_PROGRES` | [`services/simulator/app.py`](../services/simulator/app.py), [`services/simulator/Dockerfile`](../services/simulator/Dockerfile), [`tests/test_m1_slice.py`](../tests/test_m1_slice.py) | API create/operation lookup, healthcheck, idempotensi in-memory, dan image non-root tersedia; persistence dan fault injection belum dibuat. |

## M2 — Intake dan conversation assembly

| ID | Pekerjaan | Status | Bukti | Catatan |
|---|---|---|---|---|
| INT-01 | OpenWA dan replay connector | `BELUM_DIMULAI` | Belum ada service | Mulai dengan replay deterministik. |
| INT-02 | Persist intake non-blocking | `BELUM_DIMULAI` | Belum ada implementasi | Target p95 ≤500 ms. |
| ASM-01 | Debounce dan timer durable | `BELUM_DIMULAI` | Belum ada implementasi | Quiet window 5 detik, batas 20 detik. |
| ASM-02 | Asosiasi multi-kasus dan quoted reply | `BELUM_DIMULAI` | Belum ada implementasi | Horizon kandidat 48 jam. |

## M3 — Dataset dan intelligence lokal

| ID | Pekerjaan | Status | Bukti | Catatan |
|---|---|---|---|---|
| DAT-01 | Dataset sintetis dan split anti-kebocoran | `BELUM_DIMULAI` | Belum ada dataset/generator | Prasyarat evaluasi yang valid. |
| ML-01 | DAPT IndoBERT | `BELUM_DIMULAI` | Belum ada checkpoint | CPU FP32 menjadi baseline. |
| ML-02 | Klasifikasi multi-task | `BELUM_DIMULAI` | Belum ada kode/model | Intent, kategori, risiko, kelengkapan. |
| ML-03 | NER BIO spans | `BELUM_DIMULAI` | Belum ada kode/model | Lokasi, objek, dan waktu. |
| ML-04 | Chunking dan agregasi aduan panjang | `BELUM_DIMULAI` | Belum ada implementasi | 448 token dengan overlap 64. |

## M4 — Orkestrasi, trust, konflik, dan policy

| ID | Pekerjaan | Status | Bukti | Catatan |
|---|---|---|---|---|
| ORC-01 | Fungsi keputusan murni | `BELUM_DIMULAI` | Belum ada implementasi | Empat mode keputusan KAWAL. |
| ORC-02 | Contextual trust | `BELUM_DIMULAI` | Belum ada implementasi | Kontribusi riset utama. |
| ORC-03 | Diagnosis konflik | `BELUM_DIMULAI` | Belum ada implementasi | Tujuh kelas konflik. |
| ORC-04 | Adaptive escalation dan budget | `BELUM_DIMULAI` | Belum ada implementasi | Berbasis VOI. |
| POL-01 | Hard policy gate | `BELUM_DIMULAI` | Belum ada Rego/policy service | Fail-closed untuk egress dan eksekusi; egress gambar memakai sinyal deterministik non-visi. |

## M5 — Eksekusi dan keandalan

| ID | Pekerjaan | Status | Bukti | Catatan |
|---|---|---|---|---|
| TKT-02 | Simulator lengkap dan fault injection | `BELUM_DIMULAI` | Belum ada service | Create, update, transfer, close, status. |
| CLR-01 | Mesin klarifikasi | `BELUM_DIMULAI` | Belum ada implementasi | Maks. 2 pertanyaan/ronde, 3 ronde, 72 jam; meminta fakta teks saat gambar sensitif/ambigu tertahan policy. |
| REL-01 | Retry, DLQ, circuit breaker, rekonsiliasi | `BELUM_DIMULAI` | Belum ada implementasi | Harus menjamin nol tiket ganda. |

## M6 — Evaluasi tesis

| ID | Pekerjaan | Status | Bukti | Catatan |
|---|---|---|---|---|
| EVA-01 | Baseline B0–B4 dan ablasi A1–A5 | `BELUM_DIMULAI` | Belum ada harness | Evaluasi paired execution. |
| EVA-02 | Robustness, invariant, dan benchmark latency | `BELUM_DIMULAI` | Belum ada harness | Sertakan fault injection dan 95% CI. |
| EVA-03 | Panduan reproduksibilitas dan laporan hasil | `BELUM_DIMULAI` | Belum ada dokumen/hasil | Hanya setelah parameter dan manifest dibekukan. |

## Riwayat

| Tanggal | Perubahan | Bukti |
|---|---|---|
| 2026-09-11 | Baseline progres dan roadmap proyek dicatat. | [`plan.md`](plan.md), [`milestone.md`](milestone.md) |
| 2026-09-11 | Desain KAWAL diverifikasi; kebijakan egress gambar tanpa visi lokal dibekukan. | [PRD.md](PRD.md), [DESIGN.md](DESIGN.md), [SECURITY.md](SECURITY.md) |
| 2026-09-11 | Prototype M1 replay tiga bubble hingga receipt tiket idempoten dibuat. | [`../contracts/models.py`](../contracts/models.py), [`../services/`](../services/), [`../tests/test_m1_slice.py`](../tests/test_m1_slice.py) |
| 2026-09-11 | Fondasi Compose, skema PostgreSQL, dan bundle OPA M1 ditambahkan. | [`../docker-compose.yml`](../docker-compose.yml), [`../infra/migrations/001_m1_core_schema.sql`](../infra/migrations/001_m1_core_schema.sql), [`../policies/`](../policies/) |
| 2026-09-11 | Image Ticket Simulator dan CI Docker Hub disiapkan; Git lokal diinisialisasi pada branch `main`. | [`../services/simulator/Dockerfile`](../services/simulator/Dockerfile), [`../.github/workflows/simulator-ci.yml`](../.github/workflows/simulator-ci.yml) |
