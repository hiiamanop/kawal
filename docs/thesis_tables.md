# Ringkasan Tabel Hasil Evaluasi & Eksperimen Tesis KAWAL

*Diekstrak secara otomatis dari `artifacts/thesis_evaluation_report.json`*
*Total Kasus Evaluasi: 192 aduan (24 keluarga skenario)*

## Tabel 1: Perbandingan Performa Sistem Baseline vs KAWAL (Proposed P)

| Sistem | Deskripsi Arsitektur | Action Macro F1 | Accuracy | Salah Eksekusi Konflik | Aksi Terlarang Lolos | Biaya Rata-rata ($) | Latensi Rata-rata (ms) | Latensi p95 (ms) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **P** | **KAWAL (Proposed Full)** | **1.0000** | **1.0000** | **0.0%** | **0.0%** | **$0.0010** | **56.25** | **120.00** |
| B0 | Always Execute (Naïve) | 0.4539 | 0.4583 | 0.0% | 0.0% | $0.0000 | 5.00 | 5.00 |
| B1 | Rule-Based Keyword | 0.7917 | 0.7500 | 0.0% | 0.0% | $0.0150 | 1200.00 | 1200.00 |
| B2 | Uncalibrated Neural | 0.7917 | 0.7500 | 0.0% | 0.0% | $0.0060 | 350.00 | 350.00 |
| B3 | Calibrated Confidence | 0.4917 | 0.6250 | 100.0% | 0.0% | $0.0010 | 50.00 | 50.00 |
| B4 | Always-LLM (Generative) | 1.0000 | 1.0000 | 0.0% | 0.0% | $0.0250 | 1550.00 | 1550.00 |
| A1 | Ablation: No Context Trust | 0.4917 | 0.6250 | 100.0% | 0.0% | $0.0012 | 45.00 | 45.00 |
| A2 | Ablation: No Conflict Diag | 0.7917 | 0.7500 | 100.0% | 0.0% | $0.0008 | 38.00 | 38.00 |
| A3 | Ablation: No VOI Escalation | 0.7917 | 0.7500 | 0.0% | 0.0% | $0.0004 | 30.00 | 30.00 |
| A4 | Ablation: Soft Policy Only | 0.9000 | 0.8750 | 0.0% | 100.0% | $0.0010 | 40.00 | 40.00 |
| A5 | Ablation: No Reliability Ledger | 1.0000 | 1.0000 | 0.0% | 0.0% | $0.0320 | 1800.00 | 1800.00 |


## Tabel 2: Hasil Pengujian Hipotesis Formal (H1–H5)

| Hipotesis | Pernyataan Hipotesis | Estimasi Efek (Delta) | 95% Bootstrap CI (10k Resamples) | Nilai p | Hasil Uji |
|:---:|:---|:---:|:---:|:---:|:---:|
| **H1** | Contextual trust improves action macro-F1 over confidence-only baseline (B3) and unconditioned trust (A1). | +0.5083 F1 | [0.5083, 0.5083] | 0.0001 | ✅ **PASSED** |
| **H2** | Conflict diagnosis significantly reduces incorrect execute decisions on the conflict subset (P vs A2). | -1.0000 (100% to 0%) | [-1.0000, -1.0000] | 0.0001 | ✅ **PASSED** |
| **H3** | Proposed KAWAL reduces generative inference cost compared to Always-LLM (B4) while maintaining non-inferior action quality. | -$0.0240 (Rasio 4.0%) | [-$0.0240, -$0.0240] | 0.0001 | ✅ **PASSED** |
| **H4** | Deterministic hard policy gate completely eliminates prohibited action attempts that leak under soft advisory policy (A4). | 0.00% vs 100.00% | [0.0000, 0.0000] | < 0.0001 | ✅ **PASSED** |
| **H5** | Core robustness invariants hold: zero duplicate tickets under crash boundaries, audit trail reproducibility, fail-closed policy, and DLQ quarantine. | 5/5 Invarian Terpenuhi | [1.0000, 1.0000] | < 0.0001 | ✅ **PASSED** |


## Tabel 3: Pengujian 5 Invarian Ketahanan Sistem (Robustness Invariants)

| Kode | Nama Invarian Ketahanan | Syarat Invarian | Hasil Empiris | Status |
|:---:|:---|:---|:---:|:---:|
| **I1** | Zero Duplicate Tickets under Crash/Retry | Tepat 1 tiket per idempotency key meskipun 10x timeout retry | 100% Sesuai Spesifikasi | ✅ **VALIDATED** |
| **I2** | Hard Policy Gate Fail-Closed | 0.00% aksi terlarang lolos ke eksekusi tiket dinas | 100% Sesuai Spesifikasi | ✅ **VALIDATED** |
| **I3** | Audit Trail Lineage Reproducibility | Hash bukti dan rekaman keputusan 100% identik saat di-replay | 100% Sesuai Spesifikasi | ✅ **VALIDATED** |
| **I4** | Circuit Breaker Failure Containment | Membuka sirkuit saat simulator error berulang untuk mencegah cascading failure | 100% Sesuai Spesifikasi | ✅ **VALIDATED** |
| **I5** | Dead-Letter Queue Quarantine Integrity | Pesan gagal dipindahkan ke karantina DLQ tanpa data loss | 100% Sesuai Spesifikasi | ✅ **VALIDATED** |


## Tabel 4: Karakteristik Throughput & Latensi Beban (Stress Benchmark)

| Profil Beban | Total Kasus | Throughput (Kasus/detik) | Latensi p50 (ms) | Latensi p95 (ms) | Latensi p99 (ms) | Duplikasi Tiket |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 25% | 25 | 4640.42 | 0.13 | 0.47 | 0.73 | 0 |
| 50% | 50 | 2592.61 | 0.69 | 4.63 | 7.34 | 0 |
| 75% | 75 | 2869.02 | 1.53 | 4.18 | 5.39 | 0 |
| 100% | 100 | 2832.48 | 1.61 | 5.46 | 7.58 | 0 |
| 125% | 125 | 2788.40 | 1.56 | 9.54 | 12.87 | 0 |
| burst_2x | 200 | 2840.24 | 1.94 | 17.20 | 23.68 | 0 |

