# Rencana Implementasi KAWAL

## Tujuan P0

Membangun dan mengevaluasi orkestrator model-agnostik untuk memproses aduan publik berbahasa Indonesia dari WhatsApp. Sistem mengutamakan inferensi lokal, menaikkan komputasi secara adaptif bila diperlukan, menerapkan policy gate deterministik, dan membuat tiket pada simulator secara idempoten serta dapat diaudit.

## Batasan P0

- Satu tenant penelitian, satu akun WhatsApp uji, dan replay connector deterministik.
- Enam kategori: jalan, drainase/banjir, sampah, air bersih, adminduk, dan kesehatan/BPJS.
- Teks multibubble serta gambar JPEG, PNG, atau WebP hingga 10 MB; maksimal tiga gambar per siklus.
- Inferensi lokal CPU FP32: IndoBERT, NER, embedding retrieval, dan aturan deterministik.
- Tidak ada model visi lokal maupun redaksi gambar berbasis model lokal.
- Eskalasi remote untuk LLM/VLM hanya melalui `ModelGateway` dan setelah policy gate deterministik non-visi mengizinkan; gambar sensitif atau ambigu tetap privat dan beralih ke klarifikasi teks.
- Empat mode keputusan: `EXECUTE`, `RE_EVALUATE`, `REQUEST_CLARIFICATION`, dan `REJECT_IGNORE`.
- Ticket Simulator HTTP dengan fault injection dan rekonsiliasi idempotensi.
- Tidak termasuk integrasi pemerintah nyata, WhatsApp Business API resmi, dashboard publik, audio/video, atau human approval runtime.

## Prinsip Implementasi

- Simpan intake, snapshot, keputusan, perintah, inbox, dan outbox secara atomik serta dapat dilacak.
- Perlakukan semua input, lampiran, respons model, dan integrasi eksternal sebagai tidak tepercaya.
- Terapkan fail-closed sebelum egress remote maupun pembuatan tiket.
- Jangan menyimpan PII mentah, URL bertanda tangan, maupun secret pada log/telemetri.
- Pin versi model, policy bundle, parameter, dan artifact agar keputusan dapat direproduksi.
- Pisahkan kegagalan operasional (`WAITING_DEPENDENCY`, `BLOCKED`, `UNRESOLVED`) dari keputusan semantik.

## Milestone

### M0 — Spesifikasi dan keputusan desain

**Keluaran**

- Bekukan batas P0, taksonomi kategori, direktori yurisdiksi fiktif, dan kontrak keberhasilan eksperimen.
- Ganti atau arsipkan `docs/DESIGN.md` yang saat ini tidak relevan dengan KAWAL.
- Putuskan klien `ModelGateway`, strategi idempotensi WhatsApp, serta model/parameter yang dipin.
- Bekukan kebijakan gambar: tanpa visi lokal; gate deterministik non-visi menahan gambar sensitif atau ambigu di private storage dan menggunakan klarifikasi teks.

**Kriteria penerimaan**

- Semua keputusan terbuka PRD memiliki owner, nilai yang dipilih, dan alasan.
- Desain sistem hanya mereferensikan KAWAL.

### M1 — Fondasi data dan vertical slice

**Keluaran**

- Docker Compose untuk PostgreSQL/pgvector, storage privat, dan Redpanda.
- Migrasi skema inti: raw message immutable, case, snapshot, inbox, outbox, command, audit trace, dan model/policy registry.
- Kontrak bertipe untuk envelope event, snapshot, hasil agen, keputusan, dan perintah.
- Transaction runner, outbox relay, policy skeleton, dan Ticket Simulator `POST /v1/tickets`.

**Kriteria penerimaan**

- Replay tiga bubble menghasilkan satu tiket simulator melalui alur ingest → analisis dummy → policy → command.
- Crash pada batas commit tidak menghilangkan event yang telah diterima.

### M2 — Intake dan conversation assembly

**Keluaran**

- OpenWA connector terisolasi dan replay connector offline.
- Fast-ack intake dengan persist non-blocking.
- Debounce lima detik, batas burst 20 detik, dan timer database yang durable.
- Asosiasi multi-kasus, prioritas quoted reply, serta attachment bukti terlambat dalam horizon 48 jam.

**Kriteria penerimaan**

- p95 persist intake ≤500 ms tanpa inferensi pada callback.
- Dua kasus berselang-seling dan reply terlambat terasosiasi benar tanpa tiket ganda.

### M3 — Dataset dan intelligence lokal

**Keluaran**

- Generator trajectory aduan sintetis, split anti-kebocoran, serta held-out human-written.
- DAPT IndoBERT dan fine-tuning empat kepala: intent, kategori, risiko, kelengkapan.
- NER BIO untuk lokasi, objek, dan waktu; embedding retrieval/reranker.
- Chunking 448 token dengan overlap 64 token serta agregasi aduan panjang.
- Artifact kalibrasi temperatur dan benchmark CPU FP32.

**Kriteria penerimaan**

- Audit kebocoran antar split bernilai nol.
- Fakta akhir pada aduan panjang tetap terambil dan latency serving terdokumentasi.

### M4 — Orkestrator, trust, konflik, dan policy

**Keluaran**

- Fungsi keputusan murni dengan empat mode, transisi status, cache hasil tugas, dan budget ledger.
- Contextual trust berbasis Beta posterior shrinkage, kualitas bukti, kalibrasi, dan drift.
- Diagnosis tujuh kelas konflik lintas field/modalitas.
- Pemilihan eskalasi dengan Expected Value of Information.
- Policy Gate OPA/Rego deterministik dan `ModelGateway` terstruktur, termasuk gate egress gambar non-visi yang fail-closed.

**Kriteria penerimaan**

- Trace keputusan dapat dihitung ulang dengan toleransi float ≤1e-6.
- Semua fixture aksi terlarang ditolak fail-closed.
- Tugas identik tidak dipanggil dua kali dalam satu siklus keputusan.

### M5 — Eksekusi, klarifikasi, dan keandalan

**Keluaran**

- Ticket Simulator lengkap: create, update, transfer, close, status, dan fault profile.
- Klarifikasi template-first: maksimal dua pertanyaan per ronde, tiga ronde, kedaluwarsa 72 jam.
- Worker pool terisolasi, retry berbatas, circuit breaker, DLQ, dan rekonsiliasi.
- Idempotency key stabil: `tenant:case_id:ticket:create:v1`.

**Kriteria penerimaan**

- Sepuluh retry create pada timeout-after-commit menghasilkan tepat satu tiket.
- Bukti media yang belum `VERIFIED`, sensitif, atau ambigu tidak pernah dikirim ke worker visi; kasus yang membutuhkan fakta visual tersebut memicu klarifikasi teks.

### M6 — Evaluasi dan pembuktian tesis

**Keluaran**

- Baseline B0–B4 dan ablasi A1–A5.
- Runner paired execution menggunakan manifest yang dibekukan.
- Metrik kualitas, keselamatan, biaya, dan latensi beserta paired cluster bootstrap 10.000 resample.
- Panduan reproduksibilitas dan laporan hasil eksperimen.

**Kriteria penerimaan**

- Action Macro-F1, prohibited-action rate, biaya, dan latency memiliki 95% confidence interval.
- NFR utama terbukti: nol tiket ganda dan garis keturunan audit keputusan lengkap.

## Urutan Kerja Berikutnya

1. Selesaikan M0 dan perbarui desain agar selaras dengan KAWAL.
2. Implementasikan M1 sebagai vertical slice terkecil yang dapat diuji.
3. Lanjutkan M2–M5 secara berurutan dengan tes regresi pada setiap milestone.
4. Jalankan M6 hanya pada artifact, policy, dataset split, dan parameter yang telah dibekukan.
