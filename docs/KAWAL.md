# KAWAL

## Ringkasan

**KAWAL** adalah singkatan dari **Kerangka Agen untuk Wadah Aduan Layanan**. KAWAL merupakan sistem riset untuk mengubah rangkaian aduan publik berbahasa Indonesia—terutama dari WhatsApp—menjadi tiket terstruktur yang dapat diteruskan ke simulator layanan publik.

Fokus riset KAWAL adalah *model-agnostic trust- and policy-aware intelligent orchestration with adaptive computational escalation*. Artinya, KAWAL tidak mengandalkan satu model tunggal sebagai pengambil keputusan. Ia menggabungkan aturan deterministik, model NLP lokal, retrieval, dan model remote secara selektif; lalu memilih tindakan dengan mempertimbangkan kelengkapan bukti, kewenangan, policy, kepercayaan kontekstual, konflik, anggaran, dan latensi.

KAWAL memproses **klaim aduan pelapor**, bukan membuktikan kebenaran peristiwa di dunia nyata. Policy `ALLOW` hanya menyatakan tindakan memenuhi aturan internal sistem; bukan menyatakan laporan pelapor pasti benar.

## Masalah yang Ditangani

Aduan melalui WhatsApp jarang datang sebagai formulir yang lengkap. Pelapor dapat mengirim beberapa bubble pesan, lokasi pada pesan terakhir, foto, balasan terlambat, atau informasi yang saling bertentangan. Sistem otomatis harus mampu:

- menyatukan beberapa bubble menjadi konteks kasus tanpa membuat tiket terpisah;
- mengenali aduan, kategori, risiko, lokasi, objek, serta informasi yang masih kurang;
- menemukan unit penerima yang berwenang tanpa mengarang rute;
- meminta klarifikasi singkat bila hanya pelapor yang dapat melengkapi fakta;
- memilih kapan aturan/model lokal sudah cukup dan kapan pemanggilan model remote bernilai;
- mencegah tiket ganda meski terjadi retry, timeout, atau respons eksternal hilang; dan
- menyimpan jejak keputusan yang dapat diperiksa ulang.

Pendekatan LLM end-to-end tidak cukup untuk tujuan ini. Pemanggilan generatif tanpa batas meningkatkan biaya dan latensi, dapat mengarang rute kewenangan, dan dapat mengirim data sensitif ke luar sistem. KAWAL menempatkan model sebagai pekerja dengan tugas terbatas, bukan otoritas yang bebas memilih workflow atau menjalankan side effect.

## Batas Produk P0

P0 adalah lingkup penelitian/tesis yang realistis untuk satu tenant dan satu engineer.

| Area | Batas P0 |
|---|---|
| Kanal | Satu akun WhatsApp uji melalui OpenWA pada fase berikutnya, serta replay connector deterministik. |
| Bahasa | Bahasa Indonesia. |
| Kategori | Jalan, drainase/banjir, sampah, air bersih, administrasi kependudukan, serta kesehatan/BPJS. |
| Wilayah | Direktori uji dengan sedikitnya tiga yurisdiksi fiktif. |
| Modalitas | Teks dan gambar JPEG/PNG/WebP; maksimal 10 MB per gambar dan maksimal tiga gambar per siklus. |
| Eksekusi | Ticket Simulator HTTP, bukan API pemerintah sebenarnya. |
| Inspeksi | CLI/API untuk trace dan simulator; dashboard publik tidak wajib pada P0. |

P0 **tidak** mencakup pembuktian kejadian nyata, sanksi administratif, dispatch darurat, integrasi pemerintah nyata, WhatsApp Business API resmi, audio/video, dokumen multipage, dashboard publik wajib, atau kapasitas produksi nasional.

Tidak ada *human approval gate* pada runtime. KAWAL memutuskan secara otomatis; petugas simulator menangani tiket **setelah** tiket dibuat. Peneliti, evaluator, dan policy maintainer bekerja secara offline melalui dataset, evaluasi, serta konfigurasi yang terversi—bukan dengan menyetujui setiap kasus.

## Prinsip Arsitektur

1. **Local-first.** Jalur cepat memakai aturan, database, IndoBERT, NER, dan embedding lokal pada CPU FP32. Model remote dipakai hanya bila ada alasan yang tercatat.
2. **Satu otoritas orkestrasi.** KAWAL Orchestrator adalah satu-satunya komponen yang memilih workflow. Agent hanya menerima input terstruktur dan mengembalikan hasil terstruktur.
3. **Policy sebelum side effect.** Ticket creation, pengiriman klarifikasi, dan egress remote hanya dapat terjadi setelah gate deterministik memberi `ALLOW`.
4. **PostgreSQL sebagai sumber kebenaran.** Pada target P0, Supabase PostgreSQL menyimpan state kanonik. Broker hanya mengantar event; ia tidak menjadi pemilik state kasus.
5. **At-least-once dengan idempotensi.** Inbox/outbox transaksional, optimistic locking, dan idempotency key melindungi dari redelivery, retry, serta timeout.
6. **Keputusan dapat direproduksi.** Snapshot, hash input, versi model, versi policy, parameter, hasil agent, keputusan, dan receipt perintah harus tersimpan.
7. **Kegagalan operasional bukan penolakan semantik.** Gangguan DB, broker, provider, atau policy evaluator menghasilkan status operasional seperti `WAITING_DEPENDENCY`, `BLOCKED`, atau `UNRESOLVED`; bukan keputusan palsu bahwa aduan tidak valid.

## Komponen Target P0

```text
WhatsApp / Replay
        |
        v
Intake -> Supabase PostgreSQL + bucket Supabase Storage privat -> Outbox -> Redpanda
        |                                                    |
        v                                                    v
Conversation Assembly                                  Worker pools
        |                                       (text, NER, retrieval, LLM, VLM)
        v                                                    |
Case Snapshot <--------------------------------------- AgentResult terstruktur
        |
        v
KAWAL Orchestrator -> Policy Gate -> Tool Gateway -> Ticket Simulator
        |                              |
        |                              +-> Messaging Gateway
        v
Decision trace, command receipt, audit lineage
```

### Intake dan Assembly

Intake menyimpan setiap pesan sebagai rekaman immutable lalu memberi respons cepat ke konektor. Conversation Assembly mengelompokkan burst pesan dengan quiet window lima detik dan batas burst 20 detik. Quoted reply mendapat prioritas untuk menghubungkan pesan ke kasus lama; kandidat kasus lain dievaluasi dalam horizon 48 jam.

Setelah konteks cukup dibentuk, sistem membuat snapshot kasus dengan revisi, provenance bubble, hash bukti, dan status `READY`. Pesan atau bukti yang terlambat membuat revisi baru; mereka tidak boleh menciptakan tiket kedua untuk kasus yang sama.

### Intelligence Lokal

Target jalur lokal mencakup:

- **Multi-task IndoBERT:** intent, kategori, risiko, dan kelengkapan;
- **NER:** entitas lokasi, objek, serta waktu dalam format span BIO;
- **Completeness validator:** memeriksa field wajib berdasarkan kategori;
- **Authority resolver:** memvalidasi lokasi/kategori terhadap direktori yurisdiksi terversi;
- **Duplicate retrieval:** mencari kandidat kasus terkait tanpa auto-merge lintas pelapor pada P0.

Aduan panjang tidak boleh dipotong pada 512 token pertama. Rencana P0 memakai chunk 448 token dengan overlap 64 token, mempertahankan source-message ID dan offset evidence. Maksimum awal adalah 32 chunk per pass dan 64.000 karakter per revisi.

### Agent Terbatas Kontrak

Agent tidak mempunyai kredensial broker atau ticket provider dan tidak boleh menerbitkan command kepada agent lain. Polanya adalah:

```text
run(TaskInput) -> AgentResult
```

`TaskInput` membawa referensi snapshot immutable, parameter tugas, data view yang diizinkan, versi model/konfigurasi, deadline, dan budget yang telah direservasi. Runtime wrapper memvalidasi output, mengukur biaya/latensi, menyimpan result, dan menerbitkan event hasil.

## Model Keputusan

Fungsi keputusan KAWAL menerima snapshot kasus, hasil agent, trust, konflik, field wajib, versi policy/directory, budget, dan keadaan operasional. Keluaran utamanya adalah decision record, reason code, rencana eksekusi, state berikutnya, serta command outbox.

Hanya ada empat mode keputusan semantik:

| Mode | Dipilih ketika |
|---|---|
| `EXECUTE` | Field wajib didukung bukti, rute berwenang valid, konflik tidak menghalangi, dan policy mengizinkan aksi. |
| `RE_EVALUATE` | Tugas tambahan diperkirakan memberi nilai informasi positif dalam batas budget/deadline. |
| `REQUEST_CLARIFICATION` | Ada fakta wajib yang hilang/ambigu dan hanya pelapor yang dapat melengkapinya. |
| `REJECT_IGNORE` | Bukti secara konklusif menunjukkan spam, non-complaint, atau out-of-scope. |

`WAITING_DEPENDENCY`, `BLOCKED`, dan `UNRESOLVED` adalah **processing state**, bukan mode keputusan kelima. Contohnya, policy daemon timeout harus memblokir side effect dan menjadwalkan retry; sistem tidak boleh mengubahnya menjadi `REJECT_IGNORE`.

### Empat Level Eskalasi

| Level | Resource | Contoh penggunaan |
|---|---|---|
| L0 | Rules/database | Validasi field, exact reply link, dan directory authority. |
| L1 | IndoBERT, NER, embedding | Pemahaman aduan lokal cepat. |
| L2 | Verifikasi/retrieval lokal tambahan | Konflik lokasi atau kandidat duplicate. |
| L3 | LLM/VLM remote | Ambiguitas semantik material atau bukti visual yang benar-benar dibutuhkan dan diizinkan policy. |

Eskalasi bukan kewajiban menjalankan semua level. KAWAL memilih tugas tambahan berdasarkan *Expected Value of Information* (VOI): manfaat kualitas yang diperkirakan harus lebih besar daripada penalti kesalahan, biaya, latency, bandwidth, dan antrian. Budget harus direservasi di database sebelum task dipanggil.

## Confidence, Trust, Coverage, dan Konflik

KAWAL membedakan empat konsep yang sering tercampur:

- **Model confidence:** probabilitas prediksi pada input saat ini setelah kalibrasi.
- **Contextual trust:** reliabilitas historis agent untuk tugas dan konteks tertentu, dihitung secara konservatif dengan hierarchical Beta posterior shrinkage.
- **Evidence coverage:** sejauh mana field wajib didukung evidence/provenance.
- **Risk:** tingkat urgensi dampak aduan.

Trust bukan penilaian moral pelapor. Ia menilai reliabilitas komponen, misalnya versi NER tertentu untuk kategori jalan dan gaya bahasa tertentu. Confidence tidak boleh dikalikan begitu saja dengan trust lalu disebut probabilitas terkalibrasi.

KAWAL juga menandai konflik bertipe `CLASSIFICATION`, `ENTITY`, `LOCATION`, `RISK`, `EVIDENCE`, `TEXT_IMAGE`, dan `AUTHORITY`. Konflik menghasilkan kebutuhan verifikasi atau klarifikasi; konflik tidak langsung membuktikan aduan salah.

## Policy dan Keamanan

Policy adalah hard constraint. Model confidence, trust, maupun utility tidak dapat melewatinya. Target P0 memakai OPA/Rego bundle terversi dengan hasil `ALLOW` atau `DENY`, rule ID, reason code, constraints, bundle version, dan timestamp.

| Policy | Tujuan |
|---|---|
| `POL-01` | Egress model/provider/data class berada pada allowlist. |
| `POL-02` | Authority valid untuk kategori, lokasi, dan masa berlaku. |
| `POL-03` | Semua field wajib memiliki dukungan evidence. |
| `POL-04` | Klarifikasi hanya ke chat pemilik kasus dan meminta data minimum. |
| `POL-05` | Kasus sensitif wajib menggunakan restricted visibility. |
| `POL-06` | Command memiliki decision ID, revisi terkini, scope, expiry, dan stable idempotency key. |
| `POL-07` | Attachment valid, private, dan memiliki retention class. |
| `POL-08` | Kill switch, tenant, budget, dan channel rule berlaku. |

Policy dievaluasi saat keputusan dibuat dan diperiksa ulang oleh Tool Gateway segera sebelum side effect untuk mencegah *time-of-check to time-of-use*.

### Privasi dan Gambar

P0 berjalan tanpa model visi lokal maupun redaksi gambar berbasis model lokal. Karena itu KAWAL tidak mengirim gambar ke VLM remote untuk menentukan apakah gambar itu aman.

Sebelum egress remote, gate deterministik non-visi memeriksa metadata biner/MIME, ukuran, role attachment, kategori aduan, serta flag sensitivitas dari teks/entitas. Bila gambar sensitif atau ambigu:

1. gambar tetap berada pada bucket Supabase Storage privat;
2. tidak ada VLM call atau signed URL yang dibuat;
3. gambar tidak dikirim ke broker sebagai binary/base64;
4. bila bukti visual bersifat decision-critical, KAWAL meminta deskripsi faktual melalui klarifikasi teks;
5. bila bukti hanya supplementary, text path dapat diteruskan tanpa menunggu VLM.

Jika policy mengizinkan egress, signed URL dibuat just-in-time dengan TTL awal maksimal 120 detik dan tidak disimpan pada log, event, atau database sebagai URL permanen.

### Aturan Data dan Telemetri

- Semua teks, attachment, output model, webhook, dan respons pihak ketiga adalah input tidak tepercaya.
- PII mentah, nomor telepon, token, authorization header, dan signed URL dilarang muncul pada log, error response, telemetry, atau label metrik.
- Media disimpan privat; broker hanya membawa reference, hash, MIME, dan metadata yang diizinkan.
- Akses agent dibatasi; agent tidak memiliki arbitrary shell, database, network, atau secret access.
- Sistem menerapkan deny-by-default, least privilege, schema validation, batas ukuran input, dan query terparameterisasi.

## Tiket, Idempotensi, dan Rekonsiliasi

Pembuatan tiket menggunakan idempotency key kanonik:

```text
{tenant_id}:{case_id}:ticket:create:v1
```

Untuk Ticket Simulator:

- key sama + payload hash sama mengembalikan receipt tiket yang sama;
- key sama + payload hash berbeda menghasilkan `409 Conflict`;
- bila timeout terjadi setelah commit, Tool Gateway harus memeriksa `GET /v1/operations/{idempotency_key}` sebelum membuat retry;
- receipt ticket menjadi dasar perubahan case menuju `TICKETED` dan pengiriman tracking ke pelapor.

Dengan demikian, sistem dapat memiliki delivery at-least-once di boundary jaringan, tetapi tetap menjaga satu pembuatan tiket untuk satu operasi idempoten.

## Target Evaluasi Riset

KAWAL akan dievaluasi terhadap baseline berikut:

- **B0:** rules-only;
- **B1:** single structured LLM;
- **B2:** static multi-model DAG;
- **B3:** confidence-only orchestrator;
- **B4:** always-LLM.

Ablasi mengukur dampak contextual trust, diagnosis konflik, adaptive escalation, deterministic policy, dan selective invocation. Evaluasi memakai manifest yang dibekukan, dataset sintetis dengan pemisahan family anti-kebocoran, human-written held-out set, serta paired cluster bootstrap 10.000 resample untuk confidence interval 95%.

Metrik meliputi kualitas aksi, safety/prohibited-action rate, coverage, latency, biaya, bandwidth, duplicate handling, dan reproduksibilitas audit.

## Status Implementasi Saat Ini

KAWAL saat ini berada pada **prototype M1 durable**, bukan implementasi P0 lengkap.

| Komponen | Status saat ini |
|---|---|
| Replay intake | Tersedia: fixture tiga bubble aduan jalan dibaca secara deterministik dan dapat dipersistenkan atomik. |
| Kontrak data | Tersedia: model Pydantic untuk snapshot, analysis, policy, command, dan receipt. |
| Analisis | Dummy statis untuk skenario jalan; belum memakai IndoBERT/NER. |
| Policy | Stub Python fail-closed untuk `POL-02`, `POL-03`, dan `POL-06` dengan tiga yurisdiksi fiktif. |
| Orchestrator | Fast path `EXECUTE` untuk skenario valid. Empat mode penuh belum tersedia. |
| State dan outbox | Supabase PostgreSQL menyimpan inbox, state, snapshot, decision, command, audit trace, dan outbox atomik; relay ber-lease memulihkan event setelah crash. |
| Ticket Simulator | FastAPI in-memory: create, lookup operation, idempotent replay, conflict 409, dan operasi thread-safe. |
| Tes | Tujuh tes pytest mencakup happy path, idempotensi, denial evidence, payload conflict, operation lookup, vertical slice durable, dan recovery setelah crash. |

Berkas utama prototype:

- [`contracts/models.py`](../contracts/models.py): kontrak domain Pydantic;
- [`services/intake/replay.py`](../services/intake/replay.py): replay fixture menjadi receipt tiket;
- [`services/core/policy.py`](../services/core/policy.py): policy gate prototype;
- [`services/core/orchestrator.py`](../services/core/orchestrator.py): pembentukan command tiket;
- [`services/simulator/app.py`](../services/simulator/app.py): API FastAPI simulator;
- [`services/simulator/store.py`](../services/simulator/store.py): store idempoten in-memory;
- [`tests/test_m1_slice.py`](../tests/test_m1_slice.py): verifikasi slice M1.

Prototype belum memvalidasi Supabase Local, Redpanda, inbox/outbox, OPA/Rego, OpenWA live, media lifecycle, ML lokal, ModelGateway, VLM/LLM remote, trust calculation, conflict engine, klarifikasi interaktif, fault injection lengkap, maupun evaluasi tesis. Status terperinci dicatat pada [milestone.md](milestone.md).

## Roadmap

| Milestone | Fokus |
|---|---|
| M0 | Spesifikasi, desain, dan keputusan arsitektur. |
| M1 | Fondasi data dan vertical slice replay-to-ticket. |
| M2 | OpenWA intake dan conversation assembly durable. |
| M3 | Dataset sintetis, IndoBERT, NER, retrieval, dan kalibrasi. |
| M4 | Orchestrator lengkap, trust, conflict, VOI, serta OPA policy. |
| M5 | Klarifikasi, simulator lengkap, reliability, DLQ, dan fault injection. |
| M6 | Baseline, ablasi, evaluasi statistik, dan reproducibility guide. |

Rencana detail terdapat pada [plan.md](plan.md), sementara pekerjaan yang benar-benar telah selesai atau masih berjalan dicatat pada [milestone.md](milestone.md). PRD lengkap berada pada [PRD.md](PRD.md), desain antarmuka pada [DESIGN.md](DESIGN.md), dan ketentuan keamanan pada [SECURITY.md](SECURITY.md).
