# Analisis Kebaruan dan Research Gap KAWAL

**Status:** positioning riset dan hipotesis yang harus diuji, bukan klaim hasil empiris final.  
**Tanggal:** 2026-09-14

## Klaim posisi utama

KAWAL bukan mengklaim menemukan classifier aduan pertama, model IndoBERT pertama, policy engine pertama, atau sistem ticketing idempoten pertama. Kandidat kebaruannya adalah **integrasi dan evaluasi terukur** dari:

1. pemrosesan aduan percakapan Indonesia yang multi-bubble dan dapat meminta klarifikasi;
2. *contextual reliability* konservatif berbasis hierarchical Beta posterior, kualitas bukti, kalibrasi, dan drift;
3. diagnosis konflik lintas field/modalitas yang mempengaruhi tindakan, bukan hanya label;
4. *adaptive computational escalation* berbasis Expected Value of Information (VOI), budget, dan latensi;
5. hard policy gate sebelum egress/side effect, disertai audit replay serta ketahanan at-least-once/idempotensi.

Dengan demikian, objek penelitian yang tepat adalah **policy- and trust-aware decision orchestration untuk tindakan atas aduan**, bukan sekadar akurasi klasifikasi aduan.

## Literatur relevan dan gap konservatif

| Karya | Kontribusi terverifikasi | Gap yang relevan dengan KAWAL |
|---|---|---|
| Intani et al. (2022), *Automating Public Complaint Classification Through JakLapor Channel: A Case Study of Jakarta, Indonesia* — [DOI](https://doi.org/10.1109/isc255366.2022.9922346) | Studi klasifikasi aduan publik JakLapor Jakarta memakai pembelajaran mesin. | Berfokus pada klasifikasi/routing laporan; bukan orkestrasi percakapan multi-bubble, tindakan aman, klarifikasi, ataupun idempotensi eksekusi. |
| Agustina et al. (2024), *Development of a Public Complaint Classification Model to Support E-Government Using IndoBERT* — [DOI](https://doi.org/10.1109/icoris63540.2024.10903819) | Fine-tuning IndoBERT untuk klasifikasi teks aduan e-government Indonesia. | Klasifikasi tertutup satu tahap; tidak mendefinisikan kalibrasi tindakan, trust kontekstual, konflik, atau eskalasi adaptif berbatas sumber daya. |
| Wilie et al. (2020), *IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding* — [DOI](https://doi.org/10.18653/v1/2020.aacl-main.85), [arXiv](https://arxiv.org/abs/2009.05387) | Benchmark dan sumber daya NLU Indonesia termasuk IndoBERT. | Bukan sistem keputusan aduan end-to-end; tidak menguji percakapan WhatsApp, policy side effect, atau lifecycle ticketing. |
| Guo et al. (2017), *On Calibration of Modern Neural Networks* — [arXiv](https://arxiv.org/abs/1706.04599) | Menunjukkan neural network modern dapat tidak terkalibrasi dan mengevaluasi temperature scaling. | Kalibrasi probabilitas prediksi tidak sama dengan reliability komponen menurut konteks dan tidak menghasilkan action policy. |
| FrugalGPT, Chen et al. (2023), *How to Use Large Language Models While Reducing Cost and Improving Performance* — [arXiv](https://arxiv.org/abs/2305.05176) | Routing/cascade model LLM untuk mengurangi biaya sambil menjaga performa. | Optimasi terutama model-selection/cost; bukan VOI pada evidence, authority, conflict, safety policy, atau eksekusi ticketing. |
| Shinn et al. (2023), *Reflexion: Language Agents with Verbal Reinforcement Learning* — [arXiv](https://arxiv.org/abs/2303.11366) | Language agent memakai feedback verbal/memori untuk memperbaiki trial berikutnya. | Bukan sistem high-stakes yang memisahkan agent dari side effect memakai policy gate deterministik dan ledger idempoten. |
| Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language Models* — [arXiv](https://arxiv.org/abs/2210.03629) | Menggabungkan reasoning dan action pada language model untuk menggunakan tools. | Model masih berperan sebagai pengatur langkah; KAWAL membatasi model sebagai worker terstruktur dan mempertahankan orchestrator/policy sebagai otoritas tunggal. |
| Schick et al. (2023), *Toolformer: Language Models Can Teach Themselves to Use Tools* — [arXiv](https://arxiv.org/abs/2302.04761) | Mengajarkan penggunaan API/tool oleh LLM melalui self-supervision. | Tidak mengatasi authorization, policy TOCTOU, transaksi outbox/inbox, atau persyaratan no-duplicate side effect. |
| Patil et al. (2023), *Gorilla: Large Language Model Connected with Massive APIs* — [arXiv](https://arxiv.org/abs/2305.15334) | Memperbaiki pemilihan dan pemanggilan API oleh LLM. | Correct API call bukan bukti action seharusnya diizinkan; tidak memodelkan provenance/evidence, constraint policy, atau reliability fault boundary. |
| Bai et al. (2022), *Constitutional AI: Harmlessness from AI Feedback* — [arXiv](https://arxiv.org/abs/2212.08073) | Menyelaraskan perilaku model menggunakan prinsip konstitusional/AI feedback. | Prinsip pada model berbeda dari hard constraint eksternal yang dapat direplay, diaudit, dan menahan side effect saat model gagal. |

## Gap sintesis

1. **Dari label ke action:** Banyak studi aduan berhenti pada klasifikasi kategori/instansi. Gapnya adalah keputusan apakah perlu execute, re-evaluate, clarification, atau reject-ignore dengan bukti dan konsekuensi.
2. **Dari dokumen tunggal ke percakapan:** Studi klasifikasi umumnya menganggap satu teks selesai. Gapnya adalah konteks berurutan multi-bubble, reply, bukti terlambat, dan satu kasus dapat berubah revisi.
3. **Dari confidence ke contextual trust:** Kalibrasi global tidak otomatis menyatakan suatu agent dapat diandalkan untuk task, kategori, ragam bahasa, dan risk band tertentu.
4. **Dari routing model ke routing tindakan:** LLM routing hemat biaya tidak otomatis menangani conflict, authority, data class, policy, budget ledger, atau deadline.
5. **Dari safety behaviour ke safety enforcement:** Prompt/alignment tidak menggantikan policy deterministik yang memeriksa egress, authority, mandatory evidence, dan side effect.
6. **Dari tool call ke reliable side effect:** Literature agent/tool umumnya tidak berfokus pada inbox/outbox, stable idempotency, unknown-outcome reconciliation, dan audit replay pada lifecycle ticket.

## Pemetaan gap ke solusi KAWAL

| Gap | Komponen KAWAL | Status bukti saat ini | Bukti yang masih dibutuhkan sebelum klaim tesis kuat |
|---|---|---|---|
| Action orchestration | `services/core/decision.py`, empat mode | Implementasi dan unit test tersedia | Gold action set offline; evaluasi held-out yang benar-benar independen |
| Multi-bubble + clarification | `services/intake/assembly.py`, `services/clarification/` | Simulasi/replay tersedia | Flow OpenWA akun test end-to-end; evaluasi multi-turn dan late evidence |
| Contextual trust | `estimate_contextual_trust()` | Implementasi formula dan determinism test tersedia | Statistik trust dari dev labels yang frozen; uji P vs B3/A1 pada data independen |
| Conflict-aware action | `diagnose_conflict()` | Taksonomi 7 conflict types dan test tersedia | Dataset conflict berlabel; F1 diagnosis dan incorrect-execute rate pada conflict subset |
| VOI escalation | `expected_value_of_information()` dan `decide()` | Implementasi deterministic tersedia | Utility/cost matrix dipraregistrasi; pengukuran resolution gain dan cost/latency nyata |
| Hard policy | policy/Rego dan `evaluate_egress()` | Unit/fault simulation tersedia | Evaluasi OPA live, TOCTOU recheck, attempted-bypass suite dengan trace nyata |
| Reliable side effect | inbox/outbox, simulator, Redpanda relay | PostgreSQL→Redpanda smoke test dan simulator fault test tersedia | Full worker pipeline live, crash/fault/load suite dengan database dan broker nyata |

## Kandidat pernyataan novelty yang defensibel

### C1 — Novelty arsitektural
> KAWAL mengusulkan kerangka orkestrasi tindakan aduan publik percakapan berbahasa Indonesia yang menggabungkan contextual trust, conflict diagnosis, adaptive escalation, dan policy enforcement deterministik dalam satu decision loop yang dapat direplay.

**Boleh diklaim sekarang sebagai desain/artefak sistem.** Jangan menulis “pertama di dunia” tanpa systematic review yang lebih formal.

### C2 — Novelty metodologis
> KAWAL memakai contextual trust konservatif sebagai *reliability weight* (bukan probabilitas kalibrasi) untuk memengaruhi penggabungan evidence/conflict dan pemilihan escalation.

**Baru menjadi kontribusi metodologis empiris** jika estimator, baseline global-trust, data split, dan ablation A1 diuji dengan protokol frozen.

### C3 — Novelty safety/reliability
> KAWAL memisahkan model reasoning dari otoritas side effect melalui policy gate deterministik, transactional inbox/outbox, dan idempotency reconciliation.

**Boleh sebagai kontribusi engineering.** Klaim bahwa ia lebih aman/andal harus didukung policy-fault, crash boundary, dan egress tests live.

### C4 — Novelty evaluasi
> KAWAL mengevaluasi kualitas action bersama safety, cost, latency, bandwidth, conflict subset, dan reliability—bukan hanya intent/category F1.

**Masih kandidat.** Harness sintetis yang ada belum membuktikan generalisasi karena baseline saat ini merupakan simulasi terkontrol dan tidak memakai held-out human-written data.

## Posisi yang direkomendasikan untuk tesis

Gunakan formula berikut:

> “Tesis ini menginvestigasi apakah orkestrasi local-first yang mengintegrasikan contextual trust, diagnosis konflik, adaptive escalation berbasis VOI, dan hard policy gate dapat meningkatkan kualitas tindakan serta keselamatan dan efisiensi sumber daya pada pemrosesan aduan publik percakapan berbahasa Indonesia, dibandingkan baseline classifier atau orchestrator yang lebih sederhana.”

Hindari formula berikut sampai tersedia bukti lebih kuat:

- “KAWAL adalah sistem pertama yang ...”
- “KAWAL terbukti production-ready.”
- “KAWAL terbukti mengungguli literature sebelumnya.”
- “Hasil sintetis membuktikan performa pada warga nyata.”

## Research questions dan outcome yang tepat

1. **RQ1:** Apakah contextual trust meningkatkan action quality dibanding confidence-only/global reliability?  
   **Bandingkan:** P vs B3 dan A1; action macro-F1 serta calibration/action quality per context.
2. **RQ2:** Apakah diagnosis konflik mengurangi incorrect execute tanpa kehilangan recall execute yang benar?  
   **Bandingkan:** P vs A2 pada conflict subset.
3. **RQ3:** Apakah escalation VOI menurunkan cost/latency/bandwidth sambil mempertahankan kualitas action dibanding always-LLM/call-all?  
   **Bandingkan:** P vs B4 dan A5; non-inferiority margin dipraregistrasi.
4. **RQ4:** Apakah hard policy + reliability architecture menahan prohibited side effects dan duplicate tickets pada fault boundary?  
   **Bandingkan:** P vs A4; rate prohibited attempt, duplicate ticket count, recovery latency.
5. **RQ5:** Seberapa baik pendekatan tetap robust terhadap noise percakapan, missing facts, late evidence, OOD style, provider outage, dan event redelivery?  
   **Evaluasi:** stress/fault suite terpisah, denominator disclosed.

## Bibliografi inti

1. Intani, S. M., Nasution, B. I., Aminanto, M. E., Nugraha, Y., Muchtar, N., & Kanggrawan, J. I. (2022). *Automating Public Complaint Classification Through JakLapor Channel: A Case Study of Jakarta, Indonesia.* IEEE ISC2. https://doi.org/10.1109/isc255366.2022.9922346
2. Agustina, N., Naseer, M., Gusdevi, H., & Rismayadi, D. A. (2024). *Development of a Public Complaint Classification Model to Support E-Government Using IndoBERT.* IEEE ICORIS. https://doi.org/10.1109/icoris63540.2024.10903819
3. Wilie, B., et al. (2020). *IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding.* AACL-IJCNLP. https://doi.org/10.18653/v1/2020.aacl-main.85
4. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML. https://arxiv.org/abs/1706.04599
5. Chen, L., Zaharia, M., & Zou, J. (2023). *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.* https://arxiv.org/abs/2305.05176
6. Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning.* https://arxiv.org/abs/2303.11366
7. Yao, S., et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* https://arxiv.org/abs/2210.03629
8. Schick, T., et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools.* https://arxiv.org/abs/2302.04761
9. Patil, S. G., et al. (2023). *Gorilla: Large Language Model Connected with Massive APIs.* https://arxiv.org/abs/2305.15334
10. Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* https://arxiv.org/abs/2212.08073
