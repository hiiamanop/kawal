# Milestone KAWAL

Dokumen ini adalah catatan progres resmi KAWAL. Item hanya berstatus **SELESAI** bila ada artefak dan bukti yang dapat diverifikasi di repositori.

- **Status proyek:** M3 dalam progres (fondasi M3 selesai)
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
| FND-01 | Infrastruktur lokal, migrasi, dan kontrak bertipe | `SELESAI` | [`supabase/config.toml`](../supabase/config.toml), [`supabase/migrations`](../supabase/migrations), [`contracts/models.py`](../contracts/models.py) | Supabase Local menyediakan PostgreSQL/pgvector dan bucket `attachments` privat; Compose menjalankan Redpanda, OPA, dan simulator. |
| FND-02 | Transaksi state, inbox, dan outbox | `SELESAI` | [`infra/db.py`](../infra/db.py), [`services/core/persistence.py`](../services/core/persistence.py), [`services/outbox/relay.py`](../services/outbox/relay.py) | Persistensi atomik dan relay ber-lease tersedia; uji crash setelah commit membuktikan event dapat dipulihkan. |
| FND-03 | Registry model/policy dan manifest | `SELESAI` | [`supabase/migrations/20260911000001_m1_audit_and_registry.sql`](../supabase/migrations/20260911000001_m1_audit_and_registry.sql), [`policies/.manifest`](../policies/.manifest) | Registry policy/model serta audit trace tersedia; bundle policy M1 berversi. |
| TKT-01 | Ticket Simulator minimal | `SELESAI` | [`services/simulator/app.py`](../services/simulator/app.py), [`services/simulator/Dockerfile`](../services/simulator/Dockerfile), [`tests/test_m1_durable_slice.py`](../tests/test_m1_durable_slice.py) | API idempoten, simulator thread-safe, persistence, outbox relay, dan recovery setelah crash tervalidasi. |

## M2 — Intake dan conversation assembly

| ID | Pekerjaan | Status | Bukti | Catatan |
|---|---|---|---|---|
| INT-01 | OpenWA dan replay connector | `SELESAI` | [`services/intake/connector.py`](../services/intake/connector.py), [`services/intake/openwa.py`](../services/intake/openwa.py) | Replay offline dan adapter OpenWA terisolasi memfilter pesan self-sent, receipt-only, dan group chat. |
| INT-02 | Persist intake non-blocking | `SELESAI` | [`services/intake/service.py`](../services/intake/service.py), [`services/intake/assembly.py`](../services/intake/assembly.py), [`tests/test_m2_intake_service.py`](../tests/test_m2_intake_service.py) | Callback hanya melakukan persistensi atomik tanpa inferensi; p95 persist tervalidasi di bawah 500 ms. |
| ASM-01 | Debounce dan timer durable | `SELESAI` | [`services/intake/assembly.py`](../services/intake/assembly.py), [`tests/test_m2_timer_recovery.py`](../tests/test_m2_timer_recovery.py) | Quiet window 5 detik, batas 20 detik, dan pemulihan lease worker tervalidasi. |
| ASM-02 | Asosiasi multi-kasus dan quoted reply | `SELESAI` | [`services/intake/assembly.py`](../services/intake/assembly.py), [`services/intake/attachments.py`](../services/intake/attachments.py), [`tests/test_m2_intake.py`](../tests/test_m2_intake.py) | Dua kasus berselang-seling, quoted reply, dan attachment terlambat dalam horizon 48 jam terasosiasi tanpa duplikasi. |

## M3 — Dataset dan intelligence lokal

Status: `DALAM_PROGRES` (Fondasi logika & arsitektur offline selesai; menunggu bobot terlatih & evaluasi data riil).

| ID | Pekerjaan | Status | Bukti | Catatan |
|---|---|---|---|---|
| DAT-01 | Dataset sintetis dan split anti-kebocoran | `DALAM_PROGRES` | [`services/dataset/`](../services/dataset/), [`tests/test_m3_dataset.py`](../tests/test_m3_dataset.py) | Generator sintetis multi-persona, stratifikasi family split, dan mesin audit zero-leakage ($n$-gram, identifier, label) selesai & tervalidasi. Blocked: pengumpulan & anotasi held-out human-written dataset. |
| ML-01 | DAPT IndoBERT | `DALAM_PROGRES` | [`services/ml/runtime.py`](../services/ml/runtime.py), [`docs/M3.md`](M3.md) | Kontrak manifest artifak terpin (SHA-256), offline runtime fail-safe, dan engine benchmark CPU terimplementasi. Blocked: training DAPT korpus aduan untuk bobot checkpoint FP32. |
| ML-02 | Klasifikasi multi-task | `DALAM_PROGRES` | [`services/intelligence/calibration.py`](../services/intelligence/calibration.py), [`contracts/models.py`](../contracts/models.py), [`tests/test_m3_intelligence.py`](../tests/test_m3_intelligence.py) | Kontrak 4 head (intent, kategori, risiko, kelengkapan), TemperatureCalibrator, ECE metric, dan retrieval leksikal/BM25 selesai. Blocked: trained head weights. |
| ML-03 | NER BIO spans | `DALAM_PROGRES` | [`services/intelligence/ner.py`](../services/intelligence/ner.py), [`tests/test_m3_intelligence.py`](../tests/test_m3_intelligence.py) | BIO tag parser, span confidence aggregator, span merger, dan heuristik regex fallback selesai & tervalidasi. Blocked: bobot IndoBERT token classification head. |
| ML-04 | Chunking dan agregasi aduan panjang | `SELESAI` | [`services/intelligence/chunking.py`](../services/intelligence/chunking.py), [`tests/test_m3_intelligence.py`](../tests/test_m3_intelligence.py) | Sliding window 448 token / 64 overlap, proyeksi koordinat offset global-lokal dua arah, dan pelestarian source message ID tervalidasi penuh. |
| INF-M3 | Registry artifak dan offline runtime abstraction | `SELESAI` | [`services/ml/runtime.py`](../services/ml/runtime.py), [`supabase/migrations/20260911000007_m3_model_artifacts.sql`](../supabase/migrations/20260911000007_m3_model_artifacts.sql), [`tests/test_m3_runtime.py`](../tests/test_m3_runtime.py) | Registry skema database, verifikasi SHA-256 streaming, isolasi zero-network (`allow_network=False`), unavailable-safe fallback, dan runner benchmark. Blocked: benchmark CPU riil dengan bobot checkpoint penuh. |

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
| 2026-09-11 | Fondasi M3 (synthetic dataset generator, zero-leak audit, chunking 448/64, BIO parser/retrieval/kalibrasi, registry & offline runtime) terverifikasi. M3 berstatus DALAM_PROGRES menunggu bobot terlatih, benchmark CPU riil, dan data held-out. | [`docs/M3.md`](M3.md), [`services/ml/runtime.py`](../services/ml/runtime.py), [`services/intelligence/`](../services/intelligence/), [`services/dataset/`](../services/dataset/), [`tests/test_m3_*.py`](../tests/) |
