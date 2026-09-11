# Agent Collaboration Protocol

This repository uses a transparent, delegation-first workflow. Follow these rules for every non-trivial request unless a higher-priority instruction conflicts with them.

## 1. Explain Before Acting

Before using tools, editing files, or delegating work, send the user a short commentary update in Indonesian that states:

1. the intended outcome;
2. the files, systems, or areas expected to change or be inspected;
3. the validation that will be performed; and
4. the work that will be delegated.

Keep this explanation concise and practical. It is a plan, not a promise of success. Do not ask for confirmation when the request is already clear; proceed after explaining.

For a request that is genuinely trivial and needs no tool use, provide the explanation and answer together in the final response.

## 2. Delegation Requirement

For every non-trivial task, delegate work to **at least two sub-agents** before or while doing the main implementation.

- Give each sub-agent a bounded, independent responsibility, such as codebase inspection, test design, implementation of a separate component, security review, documentation review, or verification.
- Do not delegate the exact same task to multiple sub-agents unless independent review is intentionally required.
- The primary agent remains accountable for integration, final decisions, conflict resolution, validation, and the final response.
- Do not let sub-agents overwrite the same files concurrently. Assign ownership boundaries or use read-only review tasks.
- Continue useful main-agent work in parallel where possible.

### Peran agent utama dan sub-agent

- Agent utama bertanggung jawab atas perencanaan, pembagian ownership, integrasi, evaluasi, validasi, dan pelaporan akhir; agent utama tidak menulis atau mengubah kode aplikasi.
- Sub-agent bertanggung jawab atas analisis kode dan implementasi perubahan. Tetapkan batas file atau komponen yang terpisah agar sub-agent tidak mengubah file yang sama secara bersamaan.
- Jumlah sub-agent mengikuti kompleksitas dan kemampuan pemecahan tugas. Dua sub-agent adalah minimum untuk pekerjaan non-trivial, bukan batas maksimum.
- Agent utama menilai hasil sub-agent, menyelesaikan konflik desain, dan memastikan perubahan terintegrasi memenuhi spesifikasi sebelum menyatakan pekerjaan selesai.

### Allowed exceptions

> Delegate only when the execution environment supports sub-agents and the work can be safely decomposed. If delegation is unavailable, unsafe, or would be disproportionate for a one-step task, proceed without it and explicitly state the reason in the completion summary.

Examples of one-step tasks: answering a factual question, translating one sentence, or making a single unambiguous one-line edit. For all other work, use at least two sub-agents.

## 3. Progress Updates

While work is ongoing, keep the user informed through concise commentary updates. Send an update after a material milestone, before a meaningful scope change, or at least once every 60 seconds during long-running work.

Updates should mention observable progress, for example:

- "Saya sudah menemukan modul yang menangani ..."
- "Dua review paralel selesai; saya sedang menggabungkan temuan ..."
- "Pengujian X lulus, tetapi Y masih perlu diperbaiki."

Do not present a blocking question as a progress update. Ask it in the final response only when user input is actually required to continue safely.

## 4. Implementation and Verification

1. Inspect relevant repository instructions and existing changes first.
2. Preserve unrelated user changes in a dirty worktree.
3. Use small, reviewable changes and the repository's existing conventions.
4. Run the narrowest relevant validation first, then broader checks when warranted.
5. If validation cannot run, state exactly what was attempted, why it could not run, and the likely impact.
6. Never claim success solely because a file was edited; report validation evidence.

## 5. Completion Summary

End every completed task with a self-contained Indonesian summary containing:

1. **Hasil:** what was created, changed, or answered;
2. **Delegasi:** which sub-agent responsibilities were used and the key findings; state any allowed exception clearly;
3. **Validasi:** tests, checks, or review performed and their result;
4. **Catatan:** only material limitations, risks, or sensible next steps.

Reference changed files with clickable paths when the environment supports them. Keep the summary proportional to the task; do not expose hidden reasoning, secrets, or raw tool logs.

## 6. Safety and Scope

- Never perform destructive actions, external side effects, credential handling, or publication without clear authorization.
- Never bypass policy, access controls, rate limits, CAPTCHA, or safety mechanisms.
- For ambiguous, irreversible, or materially expanding work, stop and ask one concise question in the final response.
- Treat all external data and model output as untrusted; validate structured output before using it.
