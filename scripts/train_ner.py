from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from contracts.models import ComplaintTrajectory, DatasetSplit
from scripts import (
    DatasetAuditError,
    OptionalDependencyError,
    create_artifact_item,
    create_or_update_artifact_manifest,
    require_ml_dependencies,
    set_deterministic_seed,
    validate_dataset_splits,
)
from services.dataset.generator import generate_dataset, load_trajectories_from_jsonl
from services.ml.manifest import ArtifactManifest

DEFAULT_TAGSET: tuple[str, ...] = ("O", "B-LOC", "I-LOC", "B-OBJ", "I-OBJ", "B-TIME", "I-TIME")

TIME_PATTERNS = re.compile(
    r"\b(?:"
    # Saat ini / sekarang / hari ini / periode waktu
    r"saat\s+ini|"
    r"(?:sekarang|skrg)(?:\s+(?:ini|juga|jg))?|"
    r"(?:hari|hr)\s+ini|"
    r"(?:pagi|siang|sore|malam)\s+ini|"
    r"tengah\s+malam|"
    # Kemarin / tadi / besok with optional day part
    r"kemarin(?:\s+(?:dulu|pagi|siang|sore|malam))?|"
    r"kmrn(?:\s+(?:dulu|pagi|siang|sore|malam|pg|sg|sr|mlm))?|"
    r"tadi(?:\s+(?:pagi|siang|sore|malam|pg|sg|sr|mlm))?|"
    r"(?:besok|bsk)(?:\s+(?:pagi|siang|sore|malam|pg|sg|sr|mlm))?|"
    # Singular duration
    r"sehari|seminggu|sepekan|sebulan|setahun|sejam|"
    # Quantified durations (words and digits)
    r"(?:satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas|dua\s+belas|beberapa|tiap|setiap)\s+(?:hari|minggu|pekan|bulan|jam|menit|detik|tahun)|"
    r"\d+\s+(?:hari|minggu|pekan|bulan|jam|menit|detik|tahun|hr|bln|thn|mgg|dtk)|"
    # Relative past/future intervals
    r"(?:minggu|pekan|bulan|tahun)\s+(?:lalu|depan)|"
    # Clock / hours / dates
    r"pukul\s+\d{1,2}(?:[:.]\d{2})?|"
    r"jam\s+\d{1,2}(?:[:.]\d{2})?|"
    r"jam\s+operasional|"
    r"jam\s+sibuk(?:\s+(?:pagi|siang|sore|malam))?|"
    r"(?:tgl|tanggal)\s+\d{1,2}(?:[/-]\d{1,2}(?:[/-]\d{2,4})?)?"
    r")\b",
    re.IGNORECASE,
)

DAMAGE_PATTERNS = re.compile(
    r"\b(?:jalan berlubang|aspal ambles|aspal amblas|badan jalan ambrol|gorong-gorong ambrol|"
    r"tumpukan sampah liar|pembakaran sampah|sampah menumpuk|saluran drainase mampet|"
    r"air limbah meluber|pipa bocor|tiang listrik miring|lampu jalan mati|trotoar patah|"
    r"jembatan retak|genangan air|bau busuk|limbah beracun)\b",
    re.IGNORECASE,
)

COMPACT_OBJECT_PATTERNS = re.compile(
    r"\b(?:"
    # Road / Transport infrastructure
    r"(?:paving\s+)?trotoar(?:\s+(?:hancur|patah|rusak|rsk))?|"
    r"(?:jalan|jln)\s+berlubang(?:\s+parah)?|"
    r"(?:jalan|jln)\s+bergelombang|"
    r"(?:jalan|jln)\s+(?:rusak|rsk)|"
    r"aspal\s+(?:ambles|amblas)|"
    r"bahu\s+aspal(?:\s+amblas)?|"
    r"badan\s+(?:jalan|jln)(?:\s+ambrol)?|"
    r"(?:akses\s+)?jembatan(?:\s+(?:penghubung|flyover|baru))?|"
    r"jembatan\s+retak|"
    r"penerangan\s+(?:jalan|jln)(?:\s+umum)?|"
    r"lampu\s+(?:jalan|jln)(?:\s+mati)?|"
    r"lampu\s+(?:pengatur\s+)?lalu\s+lintas(?:\s+padam)?|"
    r"penutup\s+besi(?:\s+lubang\s+(?:jalan|jln))?|"
    r"penutup\s+lubang(?:\s+(?:jalan|jln))?|"
    r"lubang\s+(?:jalan|jln)|"
    r"pagar\s+pengaman(?:\s+guardrail(?:\s+tol)?)?|"
    r"guardrail(?:\s+tol)?|"
    r"(?:garis\s+penyeberangan\s+)?zebra\s+cross|"
    r"separator\s+beton(?:\s+pembatas\s+(?:jalan|jln))?|"
    r"pembatas\s+(?:jalan|jln)|"
    r"celah\s+ekspansi(?:\s+jembatan(?:\s+flyover)?)?|"
    r"jalan\s+layang|flyover|"
    r"papan\s+plang(?:\s+petunjuk\s+(?:jalan|jln))?|"
    r"rambu\s+lalu\s+lintas|"
    r"tiang\s+listrik(?:\s+miring)?|"
    r"tiang\s+telepon(?:\s+roboh)?|"
    # Drainage / Flood / Water infrastructure
    r"pintu\s+air(?:\s+pengendali\s+banjir)?|"
    r"kolam\s+retensi(?:\s+pengendali\s+air\s+hujan)?|"
    r"(?:beton\s+)?gorong-gorong(?:\s+(?:saluran|tersumbat|ambrol|ambles))?|"
    r"bendungan(?:\s+penahan\s+luapan\s+air(?:\s+kali)?)?|"
    r"(?:mesin\s+)?pompa\s+drainase(?:\s+otomatis)?|"
    r"rumah\s+pompa|"
    r"timbunan\s+lumpur(?:\s+pekat)?|"
    r"parit(?:\s+comberan)?|"
    r"bak\s+kontrol(?:\s+saluran\s+pembuangan(?:\s+air)?)?|"
    r"dinding\s+plengsengan(?:\s+kali)?|"
    r"katup\s+pengendali(?:\s+pasang\s+laut)?|"
    r"air\s+rob|"
    r"sedimen\s+tanah|"
    r"tanggul(?:\s+saluran\s+air)?|"
    r"saluran\s+drainase(?:\s+mampet)?|"
    r"saluran\s+air|"
    r"luapan\s+air|"
    r"genangan\s+air|"
    r"air\s+limbah(?:\s+meluber)?|"
    # Clean Water (PDAM)
    r"pipa\s+induk(?:\s+air\s+bersih)?|"
    r"pipa(?:\s+air\s+bersih)?|"
    r"pipa\s+(?:bocor|pecah)|"
    r"semburan\s+air|"
    r"air\s+keran|aliran\s+kran\s+air|kran\s+air|kran\s+leding|"
    r"semburan\s+(?:keran|kran)(?:\s+keruh)?|"
    r"butiran\s+pasir(?:\s+kasar)?|"
    r"pasokan\s+air(?:\s+bersih)?|"
    r"debit\s+air(?:\s+pdam)?|"
    r"air\s+pdam|"
    r"sambungan\s+arloji(?:\s+meteran\s+air)?|"
    r"meteran\s+air(?:\s+pipa\s+rumah)?|"
    r"meteran\s+pelanggan(?:\s+air\s+pdam)?|"
    r"suplai\s+kran\s+leding|"
    r"instalasi\s+pipa(?:\s+distribusi\s+leding)?|"
    r"pipa\s+distribusi(?:\s+leding)?|"
    r"unit\s+pendorong\s+reservoir(?:\s+pdam)?|"
    r"reservoir\s+pdam|"
    r"distribusi\s+pipa\s+leding|"
    # Waste (Sampah / Limbah)
    r"tumpukan\s+sampah(?:\s+liar)?|"
    r"sampah\s+liar|"
    r"tempat\s+penampungan\s+sementara|tps|"
    r"pembakaran\s+sampah|"
    r"sampah\s+kabel(?:\s+dan\s+plastik)?|"
    r"sampah\s+plastik|"
    r"sampah\s+menumpuk|sampah|"
    r"limbah\s+beracun|"
    r"limbah\s+medis(?:\s+berbahaya)?|"
    r"jarum\s+suntik(?:\s+bekas)?|"
    r"sampah\s+sayur(?:\s+pasar(?:\s+tradisional)?)?|"
    r"onggokan\s+limbah\s+plastik(?:\s+kemasan\s+padat)?|"
    r"limbah\s+plastik(?:\s+kemasan\s+padat)?|"
    r"buangan\s+jeroan(?:\s+potongan\s+ternak)?|"
    r"jeroan\s+potongan\s+ternak|"
    r"rongsokan\s+limbah\s+baterai\s+elektronik|"
    r"limbah\s+baterai\s+elektronik|"
    r"aki\s+bekas|"
    r"tumpahan\s+kaleng\s+cat|"
    r"kaleng\s+cat|"
    r"pelarut\s+kimia|"
    r"wadah\s+bak\s+sampah(?:\s+seng)?|"
    r"bak\s+sampah(?:\s+seng)?|"
    r"puing\s+semen(?:\s+bekas\s+bongkaran(?:\s+gedung)?)?|"
    r"pembakaran\s+tumpukan\s+ban\s+bekas|"
    r"tumpukan\s+ban\s+bekas|ban\s+bekas|"
    r"bau\s+busuk|bau\s+menyengat|"
    # Civil Administration
    r"(?:blanko\s+)?kartu\s+tanda\s+penduduk|ktp|"
    r"surat\s+pindah(?:\s+domisili)?|"
    r"(?:sistem\s+)?antrean\s+(?:elektronik|online)|"
    r"antrean\s+membludak|"
    r"pelayanan\s+kependudukan|"
    r"pungutan\s+liar(?:\s+tidak\s+resmi)?|"
    r"pungutan\s+tidak\s+resmi|"
    r"akta\s+kelahiran|"
    r"kartu\s+keluarga|"
    r"nik(?:\s+berkas\s+kartu\s+keluarga)?|"
    r"data\s+pbi(?:\s+bantuan\s+jaminan\s+kesehatan)?|"
    r"bantuan\s+jaminan\s+kesehatan|"
    r"akta\s+kematian|"
    r"kartu\s+identitas\s+anak|"
    r"pembaruan\s+status\s+pernikahan|"
    r"status\s+pernikahan|"
    r"kendaraan\s+mobil\s+keliling(?:\s+berkas\s+kependudukan)?|"
    r"mobil\s+keliling|"
    r"biaya\s+fotokopi\s+tidak\s+resmi|"
    r"berkas\s+mutasi\s+surat\s+kependudukan|"
    r"surat\s+kependudukan|"
    # Health Services
    r"dokter\s+jaga|"
    r"puskesmas|faskes|"
    r"tenaga\s+bidan(?:\s+bersalin)?|bidan(?:\s+bersalin)?|"
    r"dokter\s+spesialis(?:\s+anak)?|"
    r"cairan\s+reagen(?:\s+uji\s+tes\s+darah)?|"
    r"loket\s+pengambilan\s+obat(?:\s+apotek)?|"
    r"(?:persediaan\s+)?tablet\s+obat(?:\s+infeksi\s+paru)?|"
    r"(?:telepon\s+)?ambulans(?:\s+gawat\s+darurat)?|"
    r"(?:stok\s+)?obat\s+rutin(?:\s+diabetes(?:\s+dan\s+darah\s+tinggi)?)?|"
    r"stok\s+obat|"
    r"depo\s+farmasi|"
    r"lemari\s+pendingin\s+vaksin|"
    r"paket\s+bantuan\s+makanan\s+gizi(?:\s+tambahan\s+balita)?|"
    r"bantuan\s+makanan\s+gizi|"
    r"ruang\s+igd|tempat\s+tidur\s+igd|igd|"
    r"alat\s+rontgen(?:\s+foto\s+rontgen)?|"
    r"foto\s+rontgen|"
    r"tabung\s+oksigen(?:\s+medis)?|"
    r"stok\s+reagen(?:\s+uji\s+laboratorium(?:\s+darah)?)?|"
    r"reagen\s+uji\s+laboratorium|"
    r"alat\s+sterilisasi(?:\s+instrumen\s+medis)?|"
    r"ruang\s+rawat\s+inap(?:\s+isolasi)?|"
    r"tandu\s+kursi\s+roda(?:\s+evakuasi\s+pasien)?|"
    r"kursi\s+roda"
    r")\b",
    re.IGNORECASE,
)

ADMIN_LOC_PATTERNS = re.compile(
    r"\b(?:"
    r"(?:Kota|Kabupaten|Kab\.|Provinsi|Prov\.|Kecamatan|Kec\.|Kelurahan|Kel\.|Desa)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|"
    r"Jakarta\s+(?:Pusat|Selatan|Barat|Timur|Utara)|"
    r"(?:Jl\.|Jalan|Jln\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"
    r")\b"
)


def _supports_kwarg(fn_or_cls: Any, kwarg_name: str) -> bool:
    try:
        import inspect

        target = fn_or_cls.__init__ if isinstance(fn_or_cls, type) else fn_or_cls
        sig = inspect.signature(target)
        return kwarg_name in sig.parameters
    except (ValueError, TypeError, AttributeError):
        return False


class NERTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name_or_path: str = Field(default="indobenchmark/indobert-base-p1", min_length=1)
    dataset_path: str | None = None
    output_dir: str = Field(default="artifacts/ner", min_length=1)
    seed: int = Field(default=42, ge=0)
    max_seq_length: int = Field(default=448, ge=16, le=512)
    learning_rate: float = Field(default=3e-5, gt=0.0)
    batch_size: int = Field(default=16, ge=1)
    gradient_accumulation_steps: int = Field(default=2, ge=1)
    num_epochs: int = Field(default=5, ge=1)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    fp16: bool = Field(default=True)
    local_files_only: bool = Field(default=True)
    dataloader_num_workers: int = Field(default=2, ge=0)
    save_total_limit: int = Field(default=2, ge=1)
    tagset: tuple[str, ...] = Field(default=DEFAULT_TAGSET)
    audit_splits: bool = True
    version: str = Field(default="v1.0.0", min_length=1)
    class_weights: dict[str, float] | None = None
    metric_for_best_model: str = Field(default="entity_f1", min_length=1)
    greater_is_better: bool = True
    early_stopping_patience: int | None = Field(default=2, ge=1)

    @field_validator("tagset")
    @classmethod
    def validate_bio_tagset(cls, tags: tuple[str, ...]) -> tuple[str, ...]:
        if not tags:
            raise ValueError("Tagset cannot be empty")
        if "O" not in tags:
            raise ValueError("Tagset must include default outside tag 'O'")
        if len(set(tags)) != len(tags):
            raise ValueError("Tagset contains duplicate tags")

        b_tags = {t[2:] for t in tags if t.startswith("B-")}
        i_tags = {t[2:] for t in tags if t.startswith("I-")}
        if not b_tags.issubset(i_tags):
            missing = b_tags - i_tags
            raise ValueError(f"Every B-tag must have corresponding I-tag. Missing: {missing}")
        return tags

    @classmethod
    def for_rtx3060(cls, dataset_path: str | None = None, **kwargs: Any) -> NERTrainConfig:
        """Create NER training configuration with safe RTX 3060 12GB FP16 defaults for ~8k trajectories."""
        defaults: dict[str, Any] = {
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "fp16": True,
            "learning_rate": 3e-5,
            "max_seq_length": 448,
            "dataloader_num_workers": 2,
            "save_total_limit": 2,
        }
        defaults.update(kwargs)
        return cls(dataset_path=dataset_path, **defaults)


def get_rtx3060_ner_config(
    dataset_path: str | None = None, **kwargs: Any
) -> NERTrainConfig:
    """Convenience helper returning RTX 3060 safe FP16 NER training config."""
    return NERTrainConfig.for_rtx3060(dataset_path=dataset_path, **kwargs)


def parse_ner_config_dict(data: dict[str, Any]) -> NERTrainConfig:
    """Parse NERTrainConfig from either flat dict or bundled M3TrainingConfig format."""
    if "hyperparameters" in data and "model_version" in data:
        hp = data.get("hyperparameters", {})
        mv = data.get("model_version", {})
        extra = hp.get("extra_params", {})
        prec = data.get("precision_config", {})
        tagset_raw = extra.get("tags", DEFAULT_TAGSET)
        tagset = tuple(tagset_raw)

        return NERTrainConfig(
            model_name_or_path=mv.get("base_model", "indobenchmark/indobert-base-p1"),
            dataset_path=extra.get("dataset_path") or data.get("dataset_path"),
            output_dir=extra.get("output_dir", "artifacts/ner"),
            seed=data.get("seed", 42),
            max_seq_length=mv.get("max_sequence_length", 448),
            learning_rate=hp.get("learning_rate", 3e-5),
            batch_size=extra.get("per_device_train_batch_size", hp.get("batch_size", 16)),
            gradient_accumulation_steps=extra.get(
                "gradient_accumulation_steps", hp.get("gradient_accumulation_steps", 2)
            ),
            num_epochs=hp.get("num_train_epochs", 5),
            warmup_ratio=hp.get("warmup_ratio", 0.1),
            weight_decay=hp.get("weight_decay", 0.01),
            fp16=extra.get("fp16", prec.get("training_precision") == "fp16"),
            local_files_only=extra.get("local_files_only", True),
            dataloader_num_workers=extra.get("dataloader_num_workers", 2),
            save_total_limit=extra.get("save_total_limit", 2),
            tagset=tagset,
            audit_splits=extra.get("audit_splits", True),
            version=mv.get("model_version", "v1.0.0"),
            class_weights=extra.get("class_weights") or hp.get("task_weights") or None,
            metric_for_best_model=extra.get("metric_for_best_model", "entity_f1"),
            greater_is_better=extra.get("greater_is_better", True),
            early_stopping_patience=extra.get("early_stopping_patience", hp.get("early_stopping_patience", 2)),
        )
    return NERTrainConfig.model_validate(data)


def parse_config_file(config_path: str | Path) -> NERTrainConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict: dict[str, Any] = json.load(f)
    return parse_ner_config_dict(cfg_dict)


def resolve_overlapping_spans(
    spans: list[tuple[int, int, str]],
    text: str | None = None,
) -> list[tuple[int, int, str]]:
    if not spans:
        return []

    # 1. Sanitize, filter invalid spans, and trim exact token boundaries if text is provided
    cleaned: list[tuple[int, int, str]] = []
    for s, e, label in spans:
        if s >= e or s < 0:
            continue
        if text is not None and e <= len(text):
            sub = text[s:e]
            stripped = sub.strip(" \t\r\n.,!?:;\"'()[]{}")
            if not stripped:
                continue
            offset = sub.find(stripped)
            s_clean = s + offset
            e_clean = s_clean + len(stripped)
            cleaned.append((s_clean, e_clean, label))
        else:
            cleaned.append((s, e, label))

    if not cleaned:
        return []

    # Priority ranking: temporal expressions and locations take precedence over objects
    prio = {"TIME": 3, "LOC": 2, "OBJ": 1}

    # Deduplicate exact boundaries: keep highest priority
    exact: dict[tuple[int, int], str] = {}
    for s, e, l in cleaned:
        if (s, e) not in exact or prio.get(l, 0) > prio.get(exact[(s, e)], 0):
            exact[(s, e)] = l

    current = sorted([(s, e, l) for (s, e), l in exact.items()], key=lambda x: (x[0], -(x[1] - x[0])))

    # 2. Resolve nested spans:
    changed = True
    while changed:
        changed = False
        new_list: list[tuple[int, int, str]] = []
        for i, out_span in enumerate(current):
            s_out, e_out, l_out = out_span
            all_inners = [
                in_span
                for j, in_span in enumerate(current)
                if i != j and s_out <= in_span[0] and in_span[1] <= e_out and (in_span[0], in_span[1]) != (s_out, e_out)
            ]
            sorted_all = sorted(all_inners, key=lambda x: x[0])
            disjoint_inners: list[tuple[int, int, str]] = []
            last_end = -1
            for inn in sorted_all:
                if inn[0] >= last_end:
                    disjoint_inners.append(inn)
                    last_end = inn[1]

            # If all disjoint inners have the same label as out_span and there are >= 2 inners:
            # Check if outer span is bridging across non-entity filler text (e.g. test 5).
            # If the gap between inners contains filler words or is large (> 6 chars), drop out_span.
            # Otherwise, out_span is a compound noun phrase (e.g. "pipa induk air bersih"), keep out_span.
            if len(disjoint_inners) >= 2 and all(inn[2] == l_out for inn in disjoint_inners):
                is_filler_bridge = False
                for k in range(len(disjoint_inners) - 1):
                    gap_s = disjoint_inners[k][1]
                    gap_e = disjoint_inners[k + 1][0]
                    gap_len = gap_e - gap_s
                    if text is not None and gap_e <= len(text):
                        gap_str = text[gap_s:gap_e].strip().lower()
                        if gap_str not in ("", "dan", "atau", "serta", "&", "yg", "yang") and gap_len > 1:
                            is_filler_bridge = True
                            break
                    elif gap_len > 6:
                        is_filler_bridge = True
                        break
                if is_filler_bridge:
                    changed = True
                    continue

            # Carve outer span ONLY around inners that have strictly higher priority:
            # Lower priority inners (e.g. OBJ inside LOC landmark) do NOT carve outer span.
            inners_to_carve = [
                inn for inn in disjoint_inners if prio.get(inn[2], 0) > prio.get(l_out, 0)
            ]
            if inners_to_carve:
                cursor = s_out
                for s_in, e_in, _ in inners_to_carve:
                    if cursor < s_in:
                        new_list.append((cursor, s_in, l_out))
                    cursor = max(cursor, e_in)
                if cursor < e_out:
                    new_list.append((cursor, e_out, l_out))
                changed = True
            else:
                new_list.append(out_span)

        exact2: dict[tuple[int, int], str] = {}
        for s, e, l in new_list:
            if s >= e:
                continue
            if text is not None and e <= len(text):
                sub = text[s:e]
                stripped = sub.strip(" \t\r\n.,!?:;\"'()[]{}")
                if not stripped:
                    continue
                offset = sub.find(stripped)
                s_c, e_c = s + offset, s + offset + len(stripped)
            else:
                s_c, e_c = s, e
            if (s_c, e_c) not in exact2 or prio.get(l, 0) > prio.get(exact2[(s_c, e_c)], 0):
                exact2[(s_c, e_c)] = l
        current = sorted([(s, e, l) for (s, e), l in exact2.items()], key=lambda x: (x[0], -(x[1] - x[0])))

    # 3. Resolve crossing overlaps and finalize non-overlapping spans
    # Priority order: higher priority label, longer span, earlier start
    sorted_candidates = sorted(current, key=lambda x: (-prio.get(x[2], 0), -(x[1] - x[0]), x[0]))
    selected: list[tuple[int, int, str]] = []
    for s, e, l in sorted_candidates:
        if not any(max(s, sel_s) < min(e, sel_e) for sel_s, sel_e, _ in selected):
            selected.append((s, e, l))

    return sorted(selected, key=lambda x: x[0])


def _pick_canonical(aliases: list[str], lower_text: str) -> str | None:
    """Return the longest alias that appears in text (token-boundary match), or None."""
    for alias in sorted(aliases, key=len, reverse=True):
        cand_lower = alias.lower().strip(" \t\r\n.,!?:;\"'()[]{}")
        if not cand_lower:
            continue
        pat = r"(?<!\w)" + re.escape(cand_lower) + r"(?!\w)"
        if re.search(pat, lower_text):
            return alias.strip(" \t\r\n.,!?:;\"'()[]{}")
    return None


def extract_grounded_object_candidates(traj: Any, text: str) -> list[str]:
    """Extract compact grounded noun-phrase object targets safely from trajectory fields.

    Each synthetic phenomenon maps to exactly one canonical compact OBJ span: the longest
    alias that appears in text with token boundaries.  Ambiguous single-word aliases that
    overlap with other phenomena (e.g. bare 'nik', bare 'sampah') are excluded so they
    cannot produce spurious OBJ gold labels.
    """
    candidates: list[str] = []
    lower_text = text.lower()

    # 1. World truth hints if present
    wt = traj.get("world_truth") if isinstance(traj, dict) else (getattr(traj, "world_truth", None) or {})
    wt = wt or {}

    # Documents (Civil Administration)
    doc = wt.get("document") or wt.get("doc")
    if doc in ("KTP", "STATUS_KTP"):
        hit = _pick_canonical(
            ["blanko kartu tanda penduduk", "kartu tanda penduduk", "status pernikahan", "data identitas", "ktp"],
            lower_text,
        )
        if hit:
            candidates.append(hit)
    elif doc == "SURAT_PINDAH":
        hit = _pick_canonical(["surat pindah domisili", "surat pindah"], lower_text)
        if hit:
            candidates.append(hit)
    elif doc == "ANTREAN_ONLINE":
        hit = _pick_canonical(["sistem antrean elektronik", "antrean elektronik", "antrean online"], lower_text)
        if hit:
            candidates.append(hit)
    elif doc == "AKTA_LAHIR":
        hit = _pick_canonical(["akta kelahiran", "pungutan tidak resmi", "pungutan liar"], lower_text)
        if hit:
            candidates.append(hit)
    elif doc == "KARTU_KELUARGA":
        hit = _pick_canonical(["kartu keluarga", "nik berkas kartu keluarga"], lower_text)
        if hit:
            candidates.append(hit)
    elif doc == "BPJS_PBI":
        hit = _pick_canonical(["data pbi bantuan jaminan kesehatan", "bantuan jaminan kesehatan"], lower_text)
        if hit:
            candidates.append(hit)
    elif doc == "AKTA_KEMATIAN":
        hit = _pick_canonical(["akta kematian"], lower_text)
        if hit:
            candidates.append(hit)
    elif doc == "KIA":
        hit = _pick_canonical(["kartu identitas anak"], lower_text)
        if hit:
            candidates.append(hit)
    elif doc == "MOBILE_ADMIN":
        hit = _pick_canonical(["kendaraan mobil keliling berkas kependudukan", "mobil keliling"], lower_text)
        if hit:
            candidates.append(hit)
    elif doc == "LEGALISIR":
        hit = _pick_canonical(["biaya fotokopi tidak resmi"], lower_text)
        if hit:
            candidates.append(hit)

    # Facilities & health services
    fac = wt.get("facility") or wt.get("fac")
    if fac == "PUSKESMAS":
        hit = _pick_canonical(["dokter jaga", "puskesmas"], lower_text)
        if hit:
            candidates.append(hit)
    elif fac == "AMBULANCE":
        hit = _pick_canonical(["telepon ambulans gawat darurat", "ambulans gawat darurat", "ambulans"], lower_text)
        if hit:
            candidates.append(hit)
    elif fac == "FARMASI":
        hit = _pick_canonical(
            ["stok obat rutin diabetes dan darah tinggi", "stok obat rutin", "obat rutin", "stok obat", "depo farmasi"],
            lower_text,
        )
        if hit:
            candidates.append(hit)
    elif fac == "COLD_CHAIN":
        hit = _pick_canonical(["lemari pendingin vaksin"], lower_text)
        if hit:
            candidates.append(hit)
    elif fac == "POSYANDU":
        hit = _pick_canonical(["paket bantuan makanan gizi tambahan balita", "bantuan makanan gizi"], lower_text)
        if hit:
            candidates.append(hit)
    elif fac == "ER_TRIAGE":
        hit = _pick_canonical(["ruang igd", "tempat tidur igd", "igd"], lower_text)
        if hit:
            candidates.append(hit)
    elif fac == "LABORATORY":
        hit = _pick_canonical(
            ["cairan reagen uji tes darah", "stok reagen uji laboratorium darah", "reagen uji laboratorium"],
            lower_text,
        )
        if hit:
            candidates.append(hit)
    elif fac == "OXYGEN":
        hit = _pick_canonical(["tabung oksigen medis", "tabung oksigen"], lower_text)
        if hit:
            candidates.append(hit)
    elif fac == "POLI_ANAK":
        hit = _pick_canonical(["alat sterilisasi instrumen medis", "dokter spesialis anak"], lower_text)
        if hit:
            candidates.append(hit)
    elif fac == "DOTS_PULMONARY":
        hit = _pick_canonical(
            ["ruang rawat inap isolasi", "persediaan tablet obat infeksi paru", "tablet obat infeksi paru"],
            lower_text,
        )
        if hit:
            candidates.append(hit)
    elif fac == "PERSALINAN":
        hit = _pick_canonical(["tenaga bidan bersalin", "tandu kursi roda evakuasi pasien", "kursi roda"], lower_text)
        if hit:
            candidates.append(hit)
    elif fac == "FARMASI_LOKET":
        hit = _pick_canonical(["loket pengambilan obat apotek", "loket farmasi obat resep", "obat resep"], lower_text)
        if hit:
            candidates.append(hit)

    # Waste management
    waste = wt.get("waste_type") or wt.get("waste")
    if waste == "TPS_OVERFLOW":
        hit = _pick_canonical(["tempat penampungan sementara", "tps"], lower_text)
        if hit:
            candidates.append(hit)
    elif waste == "ILLEGAL_DUMP":
        hit = _pick_canonical(["tumpukan sampah liar", "sampah liar"], lower_text)
        if hit:
            candidates.append(hit)
    elif waste == "OPEN_BURNING":
        hit = _pick_canonical(["pembakaran sampah", "sampah kabel dan plastik"], lower_text)
        if hit:
            candidates.append(hit)
    elif waste == "BROKEN_BIN":
        hit = _pick_canonical(["wadah bak sampah seng", "bak sampah seng", "bak sampah"], lower_text)
        if hit:
            candidates.append(hit)
    elif waste == "CONSTRUCTION_DEBRIS":
        hit = _pick_canonical(
            ["puing semen bekas bongkaran gedung", "puing semen bekas bongkaran", "puing semen"],
            lower_text,
        )
        if hit:
            candidates.append(hit)
    elif waste == "E_WASTE":
        hit = _pick_canonical(["rongsokan limbah baterai elektronik", "limbah baterai elektronik", "aki bekas"], lower_text)
        if hit:
            candidates.append(hit)
    elif waste == "HAZARDOUS_SPILL":
        hit = _pick_canonical(["tumpahan kaleng cat", "limbah beracun pelarut kimia", "pelarut kimia", "limbah beracun"], lower_text)
        if hit:
            candidates.append(hit)
    elif waste == "MARKET":
        hit = _pick_canonical(["sampah sayur pasar tradisional", "sampah sayur"], lower_text)
        if hit:
            candidates.append(hit)
    elif waste == "MEDICAL":
        hit = _pick_canonical(["limbah medis berbahaya", "jarum suntik bekas", "limbah medis"], lower_text)
        if hit:
            candidates.append(hit)
    elif waste == "PLASTIC":
        hit = _pick_canonical(
            ["onggokan limbah plastik kemasan padat", "limbah plastik kemasan padat", "limbah plastik"],
            lower_text,
        )
        if hit:
            candidates.append(hit)
    elif waste == "SLAUGHTERHOUSE":
        hit = _pick_canonical(["buangan jeroan potongan ternak", "jeroan potongan ternak", "buangan jeroan"], lower_text)
        if hit:
            candidates.append(hit)
    elif waste == "TIRE_BURNING":
        hit = _pick_canonical(["pembakaran tumpukan ban bekas", "tumpukan ban bekas", "ban bekas"], lower_text)
        if hit:
            candidates.append(hit)

    # Road infrastructure
    infra = wt.get("infra_type") or wt.get("infra")
    if infra == "ROAD":
        hit = _pick_canonical(["jalan berlubang", "jln berlubang", "aspal ambles", "aspal amblas"], lower_text)
        if hit:
            candidates.append(hit)
    elif infra == "BRIDGE_ACCESS":
        hit = _pick_canonical(
            ["badan jalan ambrol", "badan jln ambrol", "akses jembatan penghubung", "jembatan penghubung"],
            lower_text,
        )
        if hit:
            candidates.append(hit)
    elif infra == "STREET_LIGHT":
        hit = _pick_canonical(
            ["penerangan jalan umum", "penerangan jln umum", "lampu jalan", "lampu jln", "jalan bergelombang", "jln bergelombang"],
            lower_text,
        )
        if hit:
            candidates.append(hit)
    elif infra == "TRAFFIC_LIGHT":
        hit = _pick_canonical(["lampu pengatur lalu lintas", "lampu lalu lintas"], lower_text)
        if hit:
            candidates.append(hit)
    elif infra == "MANHOLE":
        hit = _pick_canonical(["penutup besi lubang jalan", "penutup lubang jalan", "lubang jalan", "lubang jln"], lower_text)
        if hit:
            candidates.append(hit)
    elif infra == "SIDEWALK":
        hit = _pick_canonical(["paving trotoar", "trotoar"], lower_text)
        if hit:
            candidates.append(hit)
    elif infra == "GUARDRAIL":
        hit = _pick_canonical(
            ["pagar pengaman guardrail tol", "pagar pengaman guardrail", "guardrail tol", "guardrail"],
            lower_text,
        )
        if hit:
            candidates.append(hit)
    elif infra == "ZEBRA_CROSS":
        hit = _pick_canonical(["garis penyeberangan zebra cross", "zebra cross"], lower_text)
        if hit:
            candidates.append(hit)
    elif infra == "SEPARATOR":
        hit = _pick_canonical(
            ["separator beton pembatas jalan", "separator beton", "pembatas jalan", "pembatas jln"],
            lower_text,
        )
        if hit:
            candidates.append(hit)
    elif infra == "EXPANSION_JOINT":
        hit = _pick_canonical(["celah ekspansi jembatan flyover", "celah ekspansi"], lower_text)
        if hit:
            candidates.append(hit)
    elif infra == "TRAFFIC_SIGN":
        hit = _pick_canonical(["papan plang petunjuk jalan", "plang petunjuk jalan", "rambu lalu lintas"], lower_text)
        if hit:
            candidates.append(hit)
    elif infra == "ROAD_SHOULDER":
        hit = _pick_canonical(["bahu aspal amblas", "bahu aspal"], lower_text)
        if hit:
            candidates.append(hit)

    # Clean water & utility
    service = wt.get("service")
    if service == "PDAM" or wt.get("category") == "CLEAN_WATER":
        hit = _pick_canonical(
            [
                "pipa induk air bersih", "pipa air bersih", "pipa distribusi leding", "pipa induk",
                "pipa bocor", "pipa pecah", "semburan air", "air keran", "pasokan air bersih",
                "debit air pdam", "sambungan arloji meteran air",
                "meteran air pipa rumah", "meteran pelanggan air pdam", "suplai kran leding",
                "instalasi pipa distribusi leding", "unit pendorong reservoir pdam",
                "reservoir pdam", "distribusi pipa leding",
            ],
            lower_text,
        )
        if hit:
            candidates.append(hit)

    # Drainage & flood control
    cat = wt.get("category")
    if cat in ("DRAINAGE_FLOOD", "DRAINAGE") or wt.get("flood_depth_cm") is not None:
        hit = _pick_canonical(
            [
                "pintu air pengendali banjir", "pintu air", "kolam retensi pengendali air hujan",
                "kolam retensi", "gorong-gorong saluran", "gorong-gorong tersumbat", "gorong-gorong ambrol",
                "gorong-gorong", "bendungan penahan luapan air kali", "bendungan",
                "mesin pompa drainase otomatis", "mesin pompa drainase", "pompa drainase", "rumah pompa",
                "parit comberan", "parit", "bak kontrol saluran pembuangan air", "bak kontrol saluran",
                "bak kontrol", "dinding plengsengan kali", "dinding plengsengan", "katup pengendali pasang laut",
                "katup pengendali", "tanggul saluran air", "tanggul", "saluran drainase mampet",
                "saluran drainase", "saluran air", "luapan air", "genangan air", "air limbah meluber",
            ],
            lower_text,
        )
        if hit:
            candidates.append(hit)

    # 2. Decompose complaint issue clause into compact noun phrases
    NON_OBJECT_WORDS = {
        "total", "tercemar", "tergeletak", "bertumpuk", "teronggok", "terbengkalai", "ditolak",
        "mangkir", "alasan", "kabar", "pengesahan", "cap", "bayar", "perih", "mata", "kondisi",
        "keadaan", "situasi", "masalah", "keluhan", "bantuan", "petugas", "warga", "pengendara",
        "pejalan", "pemotor", "pasien", "pemberitahuan", "pengurusan", "pelayanan", "keterangan",
        "informasi", "laporan", "hancur", "rusak", "rsk", "pecah", "mati", "hilang", "habis",
        "meluap", "mengendap", "melimpah", "tersendat", "tertunda", "tertahan", "penyok", "pudar",
        "terguling", "renggang", "patah", "miring", "roboh", "tergerus", "jebol", "macet",
        "bocor", "terputus", "dipersulit", "mandek", "kebun", "tanah", "lapang", "jalan raya",
        "terbakar", "tergencet", "menggantung", "terhambat", "semburan", "masuk", "membanjiri",
        "mengalir", "berbahaya", "berada", "terbuka",
    }

    issues_to_check: list[str] = []
    obs = traj.get("observable_facts") if isinstance(traj, dict) else getattr(traj, "observable_facts", ())
    if obs and len(obs) > 0:
        issues_to_check.append(obs[0])
    turns_val = traj.get("turns") if isinstance(traj, dict) else getattr(traj, "turns", ())
    for turn in turns_val:
        t_obs = turn.get("observable_facts") if isinstance(turn, dict) else getattr(turn, "observable_facts", ())
        if t_obs and len(t_obs) > 0 and t_obs[0] not in issues_to_check:
            issues_to_check.append(t_obs[0])

    clause_delims = re.compile(
        r"\b(?:di|ke|dari|dalam|pada|tanpa|saat|ketika|karena|sehingga|akibat|untuk|oleh|"
        r"dan|atau|serta|tetapi|namun|"
        r"tidak\s+berada|tidak\s+diangkat|tidak\s+diangkut|gak\s+diangkut|"
        r"tertunda|terhenti\s+total|terhenti|mati\s+total|pecah|rusak|hilang|habis|"
        r"membludak|eror|menumpuk|meluber|menimbulkan|tersumbat|retak|mampet|"
        r"ambles|amblas|ambrol|berbau\s+busuk|berbau\s+karat|tanpa\s+pemberitahuan|"
        r"padam\s+total|menyebabkan|menghalangi|membahayakan|mengakibatkan|"
        r"kosong\s+tandas|terbengkalai|keterlambatan|belum\s+hadir)\b|"
        r"[,;.:]",
        re.IGNORECASE,
    )

    seen_issue_cands: set[str] = set()
    for issue in issues_to_check:
        for m in COMPACT_OBJECT_PATTERNS.finditer(issue):
            matched_obj = m.group(0).strip(" \t\r\n.,!?:;\"'()[]{}")
            if matched_obj and matched_obj.lower() not in seen_issue_cands:
                seen_issue_cands.add(matched_obj.lower())
                candidates.append(matched_obj)

        parts = clause_delims.split(issue)
        for p in parts:
            clean = p.strip(" \t\r\n.,!?:;\"'()[]{}")
            words = [w.lower() for w in clean.split()]
            if 1 <= len(words) <= 4 and len(clean) >= 3:
                if not TIME_PATTERNS.fullmatch(clean):
                    sub_matches = list(COMPACT_OBJECT_PATTERNS.finditer(clean))
                    if sub_matches:
                        for sm in sub_matches:
                            matched_cand = sm.group(0).strip(" \t\r\n.,!?:;\"'()[]{}")
                            if matched_cand and matched_cand.lower() not in seen_issue_cands:
                                seen_issue_cands.add(matched_cand.lower())
                                candidates.append(matched_cand)
                    elif not any(w in NON_OBJECT_WORDS for w in words):
                        if not any(w in ("tidak", "bukan", "belum", "sangat", "amat", "tolong", "mohon") for w in words):
                            if clean.lower() not in seen_issue_cands:
                                seen_issue_cands.add(clean.lower())
                                candidates.append(clean)

    # 3. Ground candidates against text using strict token boundary regex
    grounded: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        cand_lower = cand.lower().strip(" \t\r\n.,!?:;\"'()[]{}")
        if not cand_lower or cand_lower in seen:
            continue
        pat = r"(?<!\w)" + re.escape(cand_lower) + r"(?!\w)"
        if re.search(pat, lower_text):
            seen.add(cand_lower)
            grounded.append(cand.strip(" \t\r\n.,!?:;\"'()[]{}"))

    return grounded


def _extract_canonical_from_obj(obj: Any) -> Any:
    """Extract canonical spans from dict or typed object (Pydantic model) without dropping empty sequences."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "canonical_spans" in obj and obj["canonical_spans"] is not None:
            return obj["canonical_spans"]
        if "spans" in obj and obj["spans"] is not None:
            return obj["spans"]
        if "entities" in obj and obj["entities"] is not None:
            return obj["entities"]
        meta = obj.get("metadata")
        if isinstance(meta, dict) and meta.get("canonical_spans") is not None:
            return meta.get("canonical_spans")
        return None

    if hasattr(obj, "model_fields_set"):
        fields_set = getattr(obj, "model_fields_set", set())
        if "canonical_spans" in fields_set:
            val = getattr(obj, "canonical_spans", None)
            if val is not None:
                return val
        if "spans" in fields_set:
            val = getattr(obj, "spans", None)
            if val is not None:
                return val
        if "entities" in fields_set:
            val = getattr(obj, "entities", None)
            if val is not None:
                return val
        meta = getattr(obj, "metadata", None)
        if isinstance(meta, dict) and meta.get("canonical_spans") is not None:
            return meta.get("canonical_spans")
        return None

    if hasattr(obj, "canonical_spans"):
        val = getattr(obj, "canonical_spans")
        if val is not None:
            return val
    if hasattr(obj, "spans"):
        val = getattr(obj, "spans")
        if val is not None:
            return val
    if hasattr(obj, "entities"):
        val = getattr(obj, "entities")
        if val is not None:
            return val
    meta = getattr(obj, "metadata", None)
    if isinstance(meta, dict) and meta.get("canonical_spans") is not None:
        return meta.get("canonical_spans")
    return None


def _normalize_canonical_spans(raw_spans: Any) -> list[tuple[int, int, str]]:
    """Normalize canonical spans from various formats (tuple, list, dict, Pydantic objects) to (start, end, label)."""
    if raw_spans is None:
        return []
    normalized: list[tuple[int, int, str]] = []
    for item in raw_spans:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            s, e, l = item[0], item[1], item[2]
        elif isinstance(item, dict):
            s = item.get("start") if "start" in item else item.get("start_char", item.get("start_offset"))
            e = item.get("end") if "end" in item else item.get("end_char", item.get("end_offset"))
            l = item.get("label") or item.get("type") or item.get("entity") or item.get("entity_type")
        elif hasattr(item, "start_char") or hasattr(item, "start"):
            s = getattr(item, "start_char", getattr(item, "start", 0))
            e = getattr(item, "end_char", getattr(item, "end", 0))
            l = getattr(item, "label", getattr(item, "type", getattr(item, "entity", None)))
        else:
            continue
        try:
            start_i = int(s)
            end_i = int(e)
            label_s = str(l).strip()
            if label_s.startswith(("B-", "I-")):
                label_s = label_s[2:]
            if start_i < end_i and start_i >= 0 and label_s:
                normalized.append((start_i, end_i, label_s))
        except (ValueError, TypeError):
            continue
    return normalized


def _is_ood_record(traj: Any) -> bool:
    is_d = isinstance(traj, dict)
    scen = str(traj.get("scenario_id", "") if is_d else getattr(traj, "scenario_id", "")).lower()
    fam = str(traj.get("family_id", "") if is_d else getattr(traj, "family_id", "")).lower()
    wt = traj.get("world_truth", {}) if is_d else getattr(traj, "world_truth", {})
    if not isinstance(wt, dict):
        wt = dict(wt) if hasattr(wt, "__dict__") else {}
    prov = str(
        (traj.get("provenance") if is_d else getattr(traj, "provenance", None))
        or wt.get("provenance", "")
        or wt.get("dataset_origin", "")
        or (traj.get("metadata", {}).get("provenance") if is_d else getattr(traj, "metadata", {}).get("provenance", ""))
    ).strip().lower()
    task = str((traj.get("task") if is_d else getattr(traj, "task", None)) or wt.get("task", "")).strip().lower()
    is_ood_flag = bool(
        (traj.get("is_ood") if is_d else getattr(traj, "is_ood", False))
        or wt.get("is_ood", False)
        or wt.get("ood", False)
    )
    split_val = traj.get("split") if is_d else getattr(traj, "split", None)
    split_str = (split_val.value if hasattr(split_val, "value") else str(split_val or "")).lower()

    if is_ood_flag or split_str == "ood" or task in ("holdout", "ood", "ood_canary"):
        return True
    if prov in ("synthetic_ood", "synthetic_canary"):
        return True
    if scen.startswith(("ood_", "ood-", "canary_ood", "ood_canary")) or "ood_canary" in scen:
        return True
    if fam.startswith(("fam_ood", "fam-ood", "ood_canary")) or "fam-canary" in fam or "fam_canary" in fam:
        return True
    return False


def _preserve_canonical_spans_in_record(item: dict[str, Any]) -> dict[str, Any]:
    """Preserve canonical_spans in record and world_truth before Pydantic extra-forbid conversion."""
    if not isinstance(item, dict):
        return item

    wt = item.get("world_truth")
    if not isinstance(wt, dict):
        wt = dict(wt) if hasattr(wt, "__dict__") else {}
        item["world_truth"] = wt

    top_spans = _extract_canonical_from_obj(item)
    if top_spans is not None and "canonical_spans" not in wt:
        wt["canonical_spans"] = top_spans

    bubble_map: dict[str, Any] = dict(wt.get("bubble_canonical_spans") or {})
    turns = item.get("turns", []) if isinstance(item, dict) else getattr(item, "turns", ())
    if isinstance(turns, (list, tuple)):
        for t_idx, turn in enumerate(turns):
            bubbles = turn.get("bubbles", []) if isinstance(turn, dict) else getattr(turn, "bubbles", ())
            if isinstance(bubbles, (list, tuple)):
                for b_idx, bubble in enumerate(bubbles):
                    b_spans = _extract_canonical_from_obj(bubble)
                    if b_spans is not None:
                        msg_id = str((bubble.get("source_message_id") if isinstance(bubble, dict) else getattr(bubble, "source_message_id", None)) or "")
                        if msg_id:
                            bubble_map[msg_id] = b_spans
                        bubble_map[f"{t_idx}_{b_idx}"] = b_spans

    if bubble_map:
        wt["bubble_canonical_spans"] = bubble_map
        if len(bubble_map) == 1:
            if "canonical_spans" not in wt:
                wt["canonical_spans"] = next(iter(bubble_map.values()))
        else:
            # When multiple bubbles exist, canonical_spans for full text must use projected offsets
            projected_spans: list[tuple[int, int, str]] = []
            cur_full_off = 0
            for t_idx, turn in enumerate(turns):
                bubbles = turn.get("bubbles", []) if isinstance(turn, dict) else getattr(turn, "bubbles", ())
                cur_turn_off = 0
                bubble_texts: list[str] = []
                turn_spans: list[tuple[int, tuple[int, int, str]]] = []
                for b_idx, bubble in enumerate(bubbles):
                    b_text = (bubble.get("text", "") if isinstance(bubble, dict) else getattr(bubble, "text", "")).strip()
                    if not b_text:
                        continue
                    bubble_texts.append(b_text)
                    msg_id = str((bubble.get("source_message_id") if isinstance(bubble, dict) else getattr(bubble, "source_message_id", None)) or "")
                    b_sp = bubble_map.get(msg_id) if msg_id in bubble_map else bubble_map.get(f"{t_idx}_{b_idx}")
                    if b_sp is not None:
                        for s in _normalize_canonical_spans(b_sp):
                            turn_spans.append((cur_turn_off, s))
                    cur_turn_off += len(b_text) + 1
                if bubble_texts:
                    comb = " ".join(bubble_texts)
                    for b_off, (s_s, s_e, s_lbl) in turn_spans:
                        projected_spans.append((cur_full_off + b_off + s_s, cur_full_off + b_off + s_e, s_lbl))
                    cur_full_off += len(comb) + 1
            wt["canonical_spans"] = projected_spans

    return item


def _extract_spans_for_text(
    text: str,
    traj: Any,
    expanded_loc_targets: Sequence[str],
    canonical_spans: Sequence[Any] | None = None,
) -> list[tuple[int, int, str]]:
    if canonical_spans is not None:
        norm_canonical = _normalize_canonical_spans(canonical_spans)
        return resolve_overlapping_spans(norm_canonical, text=text)

    raw_spans: list[tuple[int, int, str]] = []
    lower_text = text.lower()

    # 1. TIME patterns
    for match in TIME_PATTERNS.finditer(text):
        raw_spans.append((match.start(), match.end(), "TIME"))

    # 2. Administrative location patterns
    for match in ADMIN_LOC_PATTERNS.finditer(text):
        raw_spans.append((match.start(), match.end(), "LOC"))

    # 3. Grounded trajectory location targets
    for loc in expanded_loc_targets:
        clean_loc = loc.strip(" \t\r\n.,!?:;\"'()[]{}")
        if len(clean_loc) < 3:
            continue
        pat = r"(?<!\w)" + re.escape(clean_loc.lower()) + r"(?!\w)"
        for match in re.finditer(pat, lower_text):
            raw_spans.append((match.start(), match.end(), "LOC"))

    # 4. Compact domain object patterns
    for match in COMPACT_OBJECT_PATTERNS.finditer(text):
        raw_spans.append((match.start(), match.end(), "OBJ"))

    # 5. Dynamic grounded object candidates from trajectory facts / world_truth
    obj_cands = extract_grounded_object_candidates(traj, text)
    for cand in obj_cands:
        clean_cand = cand.strip(" \t\r\n.,!?:;\"'()[]{}")
        if len(clean_cand) < 3:
            continue
        pat = r"(?<!\w)" + re.escape(clean_cand.lower()) + r"(?!\w)"
        for match in re.finditer(pat, lower_text):
            raw_spans.append((match.start(), match.end(), "OBJ"))

    # 6. Resolve overlapping and nested spans with text-aware boundary trimming
    return resolve_overlapping_spans(raw_spans, text=text)


def extract_trajectory_ner_samples(
    trajectories: Sequence[Any],
    granularity: str | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for traj in trajectories:
        is_dict = isinstance(traj, dict)
        scenario_id = traj.get("scenario_id", "") if is_dict else getattr(traj, "scenario_id", "")
        family_id = traj.get("family_id", "") if is_dict else getattr(traj, "family_id", "")
        split_val = traj.get("split", "train") if is_dict else getattr(traj, "split", "train")
        split_str = split_val.value if hasattr(split_val, "value") else str(split_val).lower()

        if _is_ood_record(traj) and split_str in ("train", "dev", "calibration"):
            raise DatasetAuditError(
                f"OOD training contamination detected: OOD canary scenario '{scenario_id}' "
                f"(family '{family_id}') cannot be included in '{split_str}' split."
            )

        wt = traj.get("world_truth", {}) if is_dict else getattr(traj, "world_truth", {})
        if not isinstance(wt, dict):
            wt = dict(wt) if hasattr(wt, "__dict__") else {}

        prov = str(
            (traj.get("provenance") if is_dict else getattr(traj, "provenance", None))
            or wt.get("provenance", "")
            or wt.get("dataset_origin", "")
            or (traj.get("metadata", {}).get("provenance") if is_dict else getattr(traj, "metadata", {}).get("provenance", ""))
        ).strip()

        loc_targets: list[str] = []
        obs = traj.get("observable_facts") if is_dict else getattr(traj, "observable_facts", ())
        hid = traj.get("hidden_facts") if is_dict else getattr(traj, "hidden_facts", ())
        if obs and len(obs) > 1:
            loc_targets.extend(obs[1:])
        if hid:
            loc_targets.extend(hid)

        turns_data = traj.get("turns", []) if is_dict else getattr(traj, "turns", ())
        for turn in turns_data:
            t_obs = turn.get("observable_facts") if isinstance(turn, dict) else getattr(turn, "observable_facts", ())
            t_hid = turn.get("hidden_facts") if isinstance(turn, dict) else getattr(turn, "hidden_facts", ())
            if t_obs and len(t_obs) > 1:
                loc_targets.extend(t_obs[1:])
            if t_hid:
                loc_targets.extend(t_hid)

        expanded_loc_targets: list[str] = []
        for loc in loc_targets:
            clean_loc = loc.strip(" \t\r\n.,!?:;\"'()[]{}")
            if clean_loc and clean_loc not in expanded_loc_targets:
                expanded_loc_targets.append(clean_loc)
            for part in loc.split(","):
                clean_part = part.strip(" \t\r\n.,!?:;\"'()[]{}")
                if len(clean_part) >= 4 and clean_part not in expanded_loc_targets:
                    expanded_loc_targets.append(clean_part)

        total_bubbles_count = sum(
            len(t.get("bubbles", []) if isinstance(t, dict) else getattr(t, "bubbles", ()))
            for t in turns_data
        )

        turn_texts: list[str] = []
        bubble_canonical_projected: list[tuple[int, int, str]] = []
        has_any_bubble_canonical = False
        current_full_offset = 0

        for turn_idx, turn in enumerate(turns_data):
            turn_is_dict = isinstance(turn, dict)
            bubbles = turn.get("bubbles", []) if turn_is_dict else getattr(turn, "bubbles", ())
            bubble_texts: list[str] = []
            turn_bubble_spans: list[tuple[int, tuple[int, int, str]]] = []
            current_turn_offset = 0

            for b_idx, bubble in enumerate(bubbles):
                b_is_dict = isinstance(bubble, dict)
                text = (bubble.get("text", "") if b_is_dict else getattr(bubble, "text", "")).strip()
                if not text:
                    continue
                bubble_texts.append(text)

                b_canonical = _extract_canonical_from_obj(bubble)

                if b_canonical is None:
                    msg_id = str((bubble.get("source_message_id") if b_is_dict else getattr(bubble, "source_message_id", None)) or "")
                    b_map = (
                        wt.get("bubble_canonical_spans")
                        or wt.get("_bubble_canonical_spans")
                        or wt.get("canonical_spans_by_bubble")
                        or {}
                    )
                    if isinstance(b_map, dict):
                        if msg_id and msg_id in b_map:
                            b_canonical = b_map[msg_id]
                        elif f"{turn_idx}_{b_idx}" in b_map:
                            b_canonical = b_map[f"{turn_idx}_{b_idx}"]
                        elif (turn_idx, b_idx) in b_map:
                            b_canonical = b_map[(turn_idx, b_idx)]
                        elif str(b_idx) in b_map:
                            b_canonical = b_map[str(b_idx)]

                if b_canonical is None and total_bubbles_count == 1:
                    b_canonical = _extract_canonical_from_obj(traj)
                    if b_canonical is None:
                        b_canonical = wt.get("canonical_spans") if wt.get("canonical_spans") is not None else wt.get("spans")

                if b_canonical is not None:
                    has_any_bubble_canonical = True

                clean_spans = _extract_spans_for_text(
                    text, traj, expanded_loc_targets, canonical_spans=b_canonical
                )

                if b_canonical is not None or clean_spans:
                    for s_s, s_e, s_lbl in clean_spans:
                        turn_bubble_spans.append((current_turn_offset, (s_s, s_e, s_lbl)))

                sample_dict = {
                    "text": text,
                    "spans": clean_spans,
                    "canonical_spans": clean_spans,
                    "split": split_str,
                    "scenario_id": scenario_id,
                    "granularity": "bubble",
                }
                if prov:
                    sample_dict["provenance"] = prov
                samples.append(sample_dict)

                current_turn_offset += len(text) + 1

            if bubble_texts:
                combined_turn = " ".join(bubble_texts)
                for b_off, (s_s, s_e, s_lbl) in turn_bubble_spans:
                    bubble_canonical_projected.append((
                        current_full_offset + b_off + s_s,
                        current_full_offset + b_off + s_e,
                        s_lbl,
                    ))
                turn_texts.append(combined_turn)
                current_full_offset += len(combined_turn) + 1

        if not turn_texts:
            direct_text = ""
            if is_dict:
                direct_text = str(traj.get("text") or traj.get("content") or traj.get("complaint") or "").strip()
            elif hasattr(traj, "text"):
                direct_text = str(getattr(traj, "text", "")).strip()
            if direct_text:
                direct_canonical = _extract_canonical_from_obj(traj)
                if direct_canonical is None:
                    direct_canonical = wt.get("canonical_spans") if wt.get("canonical_spans") is not None else wt.get("spans")
                clean_spans = _extract_spans_for_text(
                    direct_text, traj, expanded_loc_targets, canonical_spans=direct_canonical
                )
                sample_dict = {
                    "text": direct_text,
                    "spans": clean_spans,
                    "canonical_spans": clean_spans,
                    "split": split_str,
                    "scenario_id": scenario_id,
                    "granularity": "full",
                }
                if prov:
                    sample_dict["provenance"] = prov
                samples.append(sample_dict)

        if turn_texts:
            full_text = "\n".join(turn_texts)
            if has_any_bubble_canonical or bubble_canonical_projected:
                traj_canonical = bubble_canonical_projected
            else:
                traj_canonical = _extract_canonical_from_obj(traj)
                if traj_canonical is None:
                    traj_canonical = wt.get("canonical_spans") if wt.get("canonical_spans") is not None else wt.get("spans")

            clean_spans = _extract_spans_for_text(
                full_text, traj, expanded_loc_targets, canonical_spans=traj_canonical
            )
            sample_dict = {
                "text": full_text,
                "spans": clean_spans,
                "canonical_spans": clean_spans,
                "split": split_str,
                "scenario_id": scenario_id,
                "granularity": "full",
            }
            if prov:
                sample_dict["provenance"] = prov
            samples.append(sample_dict)

    if granularity is not None:
        return [s for s in samples if s.get("granularity") == granularity]
    return samples


def align_spans_to_bio_tags(
    text: str,
    spans: list[tuple[int, int, str]],
    tagset: Sequence[str] = DEFAULT_TAGSET,
) -> tuple[list[str], list[str], list[tuple[int, int]]]:
    words: list[str] = []
    word_offsets: list[tuple[int, int]] = []

    for match in re.finditer(r"\S+", text):
        raw_word = match.group(0)
        stripped = raw_word.strip(".,!?:;\"'()[]{}")
        if not stripped:
            continue
        start_char = match.start() + raw_word.find(stripped)
        end_char = start_char + len(stripped)
        words.append(stripped)
        word_offsets.append((start_char, end_char))

    tags: list[str] = []
    prev_matched_span: tuple[int, int, str] | None = None
    tagset_set = set(tagset)

    for w_start, w_end in word_offsets:
        matched_span: tuple[int, int, str] | None = None
        for s_start, s_end, s_label in spans:
            if max(w_start, s_start) < min(w_end, s_end):
                matched_span = (s_start, s_end, s_label)
                break

        if matched_span is None:
            tags.append("O")
            prev_matched_span = None
        else:
            label = matched_span[2]
            if prev_matched_span == matched_span:
                candidate_tag = f"I-{label}"
            else:
                candidate_tag = f"B-{label}"

            if candidate_tag in tagset_set:
                tags.append(candidate_tag)
            else:
                tags.append("O")
            prev_matched_span = matched_span

    return words, tags, word_offsets


def align_spans_with_tokenizer(
    tokenizer: Any,
    text: str,
    spans: list[tuple[int, int, str]],
    label2id: dict[str, int],
    max_seq_length: int = 448,
) -> dict[str, Any]:
    encoding = tokenizer(
        text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )

    offset_mapping = encoding.get("offset_mapping") or []
    labels: list[int] = []
    prev_span: tuple[int, int, str] | None = None

    for item in offset_mapping:
        start_char, end_char = int(item[0]), int(item[1])
        if start_char == end_char:
            labels.append(-100)
            continue

        matched_span: tuple[int, int, str] | None = None
        for s_start, s_end, s_label in spans:
            if max(start_char, s_start) < min(end_char, s_end):
                matched_span = (s_start, s_end, s_label)
                break

        if matched_span is None:
            labels.append(label2id.get("O", 0))
            prev_span = None
        else:
            label = matched_span[2]
            if prev_span == matched_span:
                tag = f"I-{label}"
            else:
                tag = f"B-{label}"

            tag_id = label2id.get(tag, label2id.get("O", 0))
            labels.append(tag_id)
            prev_span = matched_span

    encoding_dict = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }
    if "token_type_ids" in encoding:
        encoding_dict["token_type_ids"] = encoding["token_type_ids"]

    return encoding_dict


def _parse_trajectory_entry(item: Any) -> Any:
    """Parse a single trajectory item, preserving canonical spans before Pydantic extra-forbid conversion."""
    if isinstance(item, ComplaintTrajectory):
        return item
    if not isinstance(item, dict):
        return item
    _preserve_canonical_spans_in_record(item)
    try:
        return ComplaintTrajectory.model_validate(item)
    except Exception:
        return item


def load_or_generate_dataset(
    dataset_path: str | None,
    seed: int,
    audit_splits: bool = True,
) -> list[Any]:
    """Load trajectories from JSONL/JSON file without schema assumptions or generate standard reproducible set."""
    if dataset_path is not None:
        p = Path(dataset_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {p}")

        trajectories: list[Any] = []
        if p.suffix.lower() == ".json":
            with p.open("r", encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("["):
                try:
                    data_list = json.loads(content)
                    if isinstance(data_list, list):
                        for item in data_list:
                            trajectories.append(_parse_trajectory_entry(item))
                except json.JSONDecodeError:
                    pass
            elif content.startswith("{"):
                try:
                    data_dict = json.loads(content)
                    if isinstance(data_dict, dict):
                        trajectories.append(_parse_trajectory_entry(data_dict))
                except json.JSONDecodeError:
                    pass
            if not trajectories:
                for line in content.splitlines():
                    line_str = line.strip()
                    if line_str:
                        try:
                            item = json.loads(line_str)
                            trajectories.append(_parse_trajectory_entry(item))
                        except json.JSONDecodeError:
                            continue
        else:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        try:
                            item = json.loads(line_str)
                            trajectories.append(_parse_trajectory_entry(item))
                        except json.JSONDecodeError:
                            continue
    else:
        trajectories = generate_dataset(num_scenarios=36, seed=seed)

    if not trajectories:
        raise ValueError("Trajectory dataset is empty.")

    if audit_splits:
        for traj in trajectories:
            is_d = isinstance(traj, dict)
            scen = str(traj.get("scenario_id", "") if is_d else getattr(traj, "scenario_id", ""))
            fam = str(traj.get("family_id", "") if is_d else getattr(traj, "family_id", ""))
            s_val = traj.get("split", "train") if is_d else getattr(traj, "split", "train")
            s_str = (s_val.value if hasattr(s_val, "value") else str(s_val)).lower()
            if _is_ood_record(traj) and s_str in ("train", "dev", "calibration"):
                raise DatasetAuditError(
                    f"OOD training contamination detected: OOD canary scenario '{scen}' "
                    f"(family '{fam}') cannot be included in '{s_str}' split."
                )
        validate_dataset_splits(trajectories, allow_violations=False)

    return trajectories


def resolve_tag_weight(class_weights: dict[str, float], tag: str) -> float:
    """Resolve class weight for a tag, supporting both exact BIO tags and stripped entity types."""
    if tag in class_weights:
        return float(class_weights[tag])
    if tag.startswith(("B-", "I-")):
        ent_type = tag[2:]
        if ent_type in class_weights:
            base_w = float(class_weights[ent_type])
            return round(base_w if tag.startswith("B-") else max(1.0, base_w * 0.8), 4)
    return 1.0


def get_weighted_ner_trainer_class(base_trainer_cls: type | None = None) -> type:
    """Return a Trainer subclass that applies class weighting in compute_loss."""
    if base_trainer_cls is None:
        try:
            from transformers import Trainer

            base_trainer_cls = Trainer
        except ImportError:
            base_trainer_cls = object

    class WeightedNERTrainer(base_trainer_cls):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any, class_weights: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **kwargs: Any,
        ) -> Any:
            labels = inputs.get("labels")
            outputs = model(**inputs)
            if self.class_weights is not None and labels is not None:
                logits = (
                    outputs.get("logits")
                    if isinstance(outputs, dict)
                    else getattr(outputs, "logits", None)
                )
                if logits is not None:
                    import torch
                    import torch.nn as nn

                    loss_fct = nn.CrossEntropyLoss(
                        weight=self.class_weights.to(logits.device),
                        ignore_index=-100,
                    )
                    loss = loss_fct(
                        logits.view(-1, logits.size(-1)), labels.view(-1)
                    )
                    if isinstance(outputs, dict):
                        outputs["loss"] = loss
                    elif hasattr(outputs, "loss"):
                        try:
                            outputs.loss = loss
                        except Exception:
                            pass
                else:
                    loss = (
                        outputs.get("loss")
                        if isinstance(outputs, dict)
                        else getattr(outputs, "loss", None)
                    )
            else:
                loss = outputs.get("loss") if isinstance(outputs, dict) else getattr(outputs, "loss", None)
            return (loss, outputs) if return_outputs else loss

    return WeightedNERTrainer


WeightedNERTrainer = get_weighted_ner_trainer_class()


def compute_ner_metrics(
    eval_pred: Any,
    id2label: dict[int, str],
) -> dict[str, float]:
    """Compute token accuracy, entity precision, recall, F1, and per-type F1 from eval predictions."""
    predictions = getattr(eval_pred, "predictions", None)
    labels = getattr(eval_pred, "label_ids", None)
    if predictions is None and isinstance(eval_pred, (tuple, list)):
        predictions, labels = eval_pred[0], eval_pred[1]
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    try:
        import numpy as np

        if hasattr(predictions, "ndim"):
            preds = np.argmax(predictions, axis=-1).tolist()
        else:
            preds = np.argmax(np.array(predictions), axis=-1).tolist()
    except Exception:
        preds = [
            [max(range(len(step)), key=lambda i: step[i]) for step in seq]
            for seq in predictions
        ]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    token_correct = 0
    token_total = 0

    per_type_counts: dict[str, dict[str, int]] = {}

    for pred_seq, label_seq in zip(preds, labels):
        seq_true_tags: list[str] = []
        seq_pred_tags: list[str] = []
        dummy_tokens: list[str] = []
        for p_idx, l_idx in zip(pred_seq, label_seq):
            if int(l_idx) == -100:
                continue
            true_t = id2label.get(int(l_idx), "O")
            pred_t = id2label.get(int(p_idx), "O")
            seq_true_tags.append(true_t)
            seq_pred_tags.append(pred_t)
            dummy_tokens.append(f"t_{token_total}")
            token_total += 1
            if true_t == pred_t:
                token_correct += 1

        if not dummy_tokens:
            continue

        try:
            from services.intelligence.ner import parse_bio_tags

            true_ents = parse_bio_tags(dummy_tokens, seq_true_tags)
            pred_ents = parse_bio_tags(dummy_tokens, seq_pred_tags)
            t_set = {(e.label, e.start_token, e.end_token) for e in true_ents}
            p_set = {(e.label, e.start_token, e.end_token) for e in pred_ents}
            tp = len(t_set.intersection(p_set))
            fp = len(p_set - t_set)
            fn = len(t_set - p_set)

            for lbl, _, _ in t_set.intersection(p_set):
                st = per_type_counts.setdefault(lbl, {"tp": 0, "fp": 0, "fn": 0})
                st["tp"] += 1
            for lbl, _, _ in p_set - t_set:
                st = per_type_counts.setdefault(lbl, {"tp": 0, "fp": 0, "fn": 0})
                st["fp"] += 1
            for lbl, _, _ in t_set - p_set:
                st = per_type_counts.setdefault(lbl, {"tp": 0, "fp": 0, "fn": 0})
                st["fn"] += 1
        except Exception:
            tp, fp, fn = 0, 0, 0

        total_tp += tp
        total_fp += fp
        total_fn += fn

    ent_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    ent_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    ent_f1 = (
        (2 * ent_prec * ent_rec) / (ent_prec + ent_rec)
        if (ent_prec + ent_rec) > 0
        else 0.0
    )
    acc = token_correct / token_total if token_total > 0 else 0.0

    metrics: dict[str, float] = {
        "entity_f1": round(ent_f1, 4),
        "ner_entity_f1": round(ent_f1, 4),
        "entity_precision": round(ent_prec, 4),
        "entity_recall": round(ent_rec, 4),
        "accuracy": round(acc, 4),
    }

    for type_name in ("LOC", "OBJ", "TIME"):
        counts = per_type_counts.get(type_name, {"tp": 0, "fp": 0, "fn": 0})
        t_tp = counts["tp"]
        t_fp = counts["fp"]
        t_fn = counts["fn"]
        p = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
        r = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        metrics[f"{type_name.lower()}_f1"] = round(f1, 4)

    return metrics


def train_transformers_ner(
    config: NERTrainConfig,
    samples: list[dict[str, Any]],
    out_dir: Path,
) -> ArtifactManifest:
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    needed_tags = set(config.tagset)
    for s in samples:
        for item in s.get("spans", []):
            if len(item) >= 3:
                ent = str(item[2]).strip()
                if ent.startswith(("B-", "I-")):
                    ent = ent[2:]
                needed_tags.add(f"B-{ent}")
                needed_tags.add(f"I-{ent}")
    seen: set[str] = set()
    effective_tagset_list: list[str] = []
    for t in config.tagset:
        if t not in seen:
            seen.add(t)
            effective_tagset_list.append(t)
    for t in sorted(needed_tags):
        if t not in seen:
            seen.add(t)
            effective_tagset_list.append(t)
    effective_tagset = tuple(effective_tagset_list)

    label2id = {tag: idx for idx, tag in enumerate(effective_tagset)}
    id2label = {idx: tag for idx, tag in enumerate(effective_tagset)}

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        local_files_only=config.local_files_only,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=len(effective_tagset),
        id2label=id2label,
        label2id=label2id,
        local_files_only=config.local_files_only,
    )

    train_samples = [
        s for s in samples if s.get("split") in (DatasetSplit.TRAIN.value, DatasetSplit.TRAIN)
    ]
    dev_samples = [
        s for s in samples if s.get("split") in (DatasetSplit.DEV.value, DatasetSplit.DEV)
    ]

    if not train_samples:
        train_samples = samples[: max(1, int(len(samples) * 0.8))]
        dev_samples = samples[len(train_samples) :]

    train_encodings = [
        align_spans_with_tokenizer(
            tokenizer,
            s["text"],
            s["spans"],
            label2id,
            max_seq_length=config.max_seq_length,
        )
        for s in train_samples
    ]
    dev_encodings = [
        align_spans_with_tokenizer(
            tokenizer,
            s["text"],
            s["spans"],
            label2id,
            max_seq_length=config.max_seq_length,
        )
        for s in dev_samples
    ]

    class TorchNERDataset(torch.utils.data.Dataset):
        def __init__(self, encodings: list[dict[str, Any]]) -> None:
            self.encodings = encodings

        def __len__(self) -> int:
            return len(self.encodings)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {k: torch.tensor(v) for k, v in self.encodings[idx].items()}

    train_dataset = TorchNERDataset(train_encodings)
    dev_dataset = TorchNERDataset(dev_encodings) if dev_encodings else None

    use_cuda = torch.cuda.is_available()
    use_fp16 = config.fp16 and use_cuda

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        return compute_ner_metrics(eval_pred, id2label)

    training_args_kwargs: dict[str, Any] = {
        "output_dir": str(out_dir),
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_epochs,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "fp16": use_fp16,
        "seed": config.seed,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": bool(dev_dataset),
        "optim": "adamw_torch",
        "dataloader_num_workers": config.dataloader_num_workers,
        "report_to": "none",
    }
    if dev_dataset:
        training_args_kwargs["metric_for_best_model"] = config.metric_for_best_model
        training_args_kwargs["greater_is_better"] = config.greater_is_better

    if _supports_kwarg(TrainingArguments, "eval_strategy"):
        training_args_kwargs["eval_strategy"] = "epoch" if dev_dataset else "no"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch" if dev_dataset else "no"

    training_args = TrainingArguments(**training_args_kwargs)

    trainer_cls = Trainer
    trainer_init_kwargs: dict[str, Any] = {}
    if config.class_weights is not None:
        weights_tensor = torch.tensor(
            [resolve_tag_weight(config.class_weights, t) for t in config.tagset],
            dtype=torch.float,
        )
        if isinstance(Trainer, type):
            trainer_cls = get_weighted_ner_trainer_class(Trainer)
            trainer_init_kwargs["class_weights"] = weights_tensor

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": data_collator,
    }
    if dev_dataset:
        trainer_kwargs["compute_metrics"] = compute_metrics

    callbacks: list[Any] = []
    if dev_dataset and config.early_stopping_patience and config.early_stopping_patience > 0:
        try:
            from transformers import EarlyStoppingCallback

            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config.early_stopping_patience
                )
            )
        except (ImportError, AttributeError):
            pass
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    if _supports_kwarg(Trainer, "processing_class"):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = trainer_cls(**trainer_kwargs, **trainer_init_kwargs)

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    best_checkpoint_path: str | None = None
    best_metric: float | None = None
    if hasattr(trainer, "state"):
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
        if isinstance(best_ckpt, (str, Path)):
            best_checkpoint_path = str(best_ckpt)
        raw_metric = getattr(trainer.state, "best_metric", None)
        if isinstance(raw_metric, (int, float)):
            best_metric = float(raw_metric)

    tagset_path = out_dir / "tagset.json"
    with tagset_path.open("w", encoding="utf-8") as f:
        json.dump(list(config.tagset), f, indent=2)

    weight_candidates = [
        out_dir / "model.safetensors",
        out_dir / "pytorch_model.bin",
    ]
    target_weights = next((p for p in weight_candidates if p.is_file()), None)
    if target_weights is None:
        try:
            from safetensors.torch import save_file

            save_file(model.state_dict(), str(out_dir / "model.safetensors"))
            target_weights = out_dir / "model.safetensors"
        except Exception:
            torch.save(model.state_dict(), str(out_dir / "pytorch_model.bin"))
            target_weights = out_dir / "pytorch_model.bin"

    config_path = out_dir / "config.json"
    if not config_path.is_file():
        if hasattr(model, "config") and hasattr(model.config, "save_pretrained"):
            model.config.save_pretrained(str(out_dir))
        elif hasattr(model, "config") and hasattr(model.config, "to_dict"):
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(model.config.to_dict(), f, indent=2)
        else:
            std_hf_config = {
                "architectures": ["BertForTokenClassification"],
                "model_type": "bert",
                "num_labels": len(config.tagset),
                "id2label": id2label,
                "label2id": label2id,
                "max_position_embeddings": config.max_seq_length,
                "hidden_size": 768,
                "vocab_size": getattr(getattr(model, "config", None), "vocab_size", 50000),
            }
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(std_hf_config, f, indent=2)
    else:
        try:
            with config_path.open("r", encoding="utf-8") as f:
                existing_cfg = json.load(f)
            if existing_cfg.get("model_type") == "IndoBERT-TokenClassification-BIO":
                existing_cfg["model_type"] = "bert"
                if "architectures" not in existing_cfg:
                    existing_cfg["architectures"] = ["BertForTokenClassification"]
                with config_path.open("w", encoding="utf-8") as f:
                    json.dump(existing_cfg, f, indent=2)
        except Exception:
            pass

    kawal_metadata_path = out_dir / "kawal_ner_metadata.json"
    kawal_meta = {
        "model_type": "IndoBERT-TokenClassification-BIO",
        "tagset": list(effective_tagset),
        "num_labels": len(effective_tagset),
        "id2label": id2label,
        "label2id": label2id,
        "max_seq_length": config.max_seq_length,
        "training_samples": len(samples),
        "fp16": config.fp16,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "class_weights": config.class_weights,
        "metric_for_best_model": config.metric_for_best_model,
        "best_model_checkpoint": best_checkpoint_path,
        "best_metric": best_metric,
    }
    with kawal_metadata_path.open("w", encoding="utf-8") as f:
        json.dump(kawal_meta, f, indent=2)

    artifact_item = create_artifact_item(
        file_path=target_weights,
        name="indobert-ner-bio",
        version=config.version,
        task="ner",
        precision="fp16" if use_fp16 else "fp32",
        relative_to=out_dir,
    )

    manifest = create_or_update_artifact_manifest(
        output_dir=out_dir,
        items=[artifact_item],
        environment="gpu-cuda" if use_cuda else "local-cpu",
    )
    return manifest


def run_train_ner(
    config: NERTrainConfig,
    dry_run: bool = False,
    validate_only: bool = False,
) -> ArtifactManifest | None:
    set_deterministic_seed(config.seed)

    trajectories = load_or_generate_dataset(
        dataset_path=config.dataset_path,
        seed=config.seed,
        audit_splits=config.audit_splits,
    )

    samples = extract_trajectory_ner_samples(trajectories)
    if not samples:
        raise ValueError("No NER training samples could be extracted from trajectories.")

    if validate_only:
        return None

    if not dry_run:
        require_ml_dependencies(
            "torch", "transformers", purpose="token classification (NER BIO) training"
        )
        out_dir = Path(config.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        return train_transformers_ner(config, samples, out_dir)

    out_dir = Path(config.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    needed_tags = set(config.tagset)
    for s in samples:
        for item in s.get("spans", []):
            if len(item) >= 3:
                ent = str(item[2]).strip()
                if ent.startswith(("B-", "I-")):
                    ent = ent[2:]
                needed_tags.add(f"B-{ent}")
                needed_tags.add(f"I-{ent}")
    seen: set[str] = set()
    effective_tagset_list: list[str] = []
    for t in config.tagset:
        if t not in seen:
            seen.add(t)
            effective_tagset_list.append(t)
    for t in sorted(needed_tags):
        if t not in seen:
            seen.add(t)
            effective_tagset_list.append(t)
    effective_tagset = tuple(effective_tagset_list)

    label2id = {tag: idx for idx, tag in enumerate(effective_tagset)}
    id2label = {idx: tag for idx, tag in enumerate(effective_tagset)}

    config_path = out_dir / "config.json"
    ner_meta = {
        "architectures": ["BertForTokenClassification"],
        "model_type": "bert",
        "base_model": config.model_name_or_path,
        "local_files_only": config.local_files_only,
        "hidden_size": 768,
        "max_position_embeddings": config.max_seq_length,
        "num_labels": len(effective_tagset),
        "label2id": label2id,
        "id2label": id2label,
        "training_samples": len(samples),
        "fp16": config.fp16,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "class_weights": config.class_weights,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": config.greater_is_better,
        "early_stopping_patience": config.early_stopping_patience,
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(ner_meta, f, indent=2)

    kawal_metadata_path = out_dir / "kawal_ner_metadata.json"
    kawal_meta = {
        "model_type": "IndoBERT-TokenClassification-BIO",
        "tagset": list(effective_tagset),
        "num_labels": len(effective_tagset),
        "id2label": id2label,
        "label2id": label2id,
        "max_seq_length": config.max_seq_length,
        "training_samples": len(samples),
        "fp16": config.fp16,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "class_weights": config.class_weights,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": config.greater_is_better,
        "early_stopping_patience": config.early_stopping_patience,
    }
    with kawal_metadata_path.open("w", encoding="utf-8") as f:
        json.dump(kawal_meta, f, indent=2)

    tagset_path = out_dir / "tagset.json"
    with tagset_path.open("w", encoding="utf-8") as f:
        json.dump(list(effective_tagset), f, indent=2)

    weight_path = out_dir / "model.safetensors"
    skeleton_payload = (
        f"KAWAL_NER_BIO_SKELETON_V1_SEED_{config.seed}_TAGS_{len(effective_tagset)}_FP16_{config.fp16}".encode(
            "utf-8"
        )
        + b"\x00" * 256
    )
    weight_path.write_bytes(skeleton_payload)

    artifact_item = create_artifact_item(
        file_path=weight_path,
        name="indobert-ner-bio",
        version=config.version,
        task="ner",
        precision="fp32",
        relative_to=out_dir,
    )

    manifest = create_or_update_artifact_manifest(
        output_dir=out_dir,
        items=[artifact_item],
        environment="local-cpu",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IndoBERT Token Classification (NER BIO) CLI & Training"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="Path to pre-trained model or cached checkpoint directory",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to JSONL dataset")
    parser.add_argument("--corpus-path", type=str, default=None, help="Alias for dataset-path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for artifacts")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--fp16", action="store_true", default=None, help="Enable fp16 training (RTX 3060)")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable fp16 training")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=None,
        help="Load local cached checkpoint files only without remote downloads",
    )
    parser.add_argument(
        "--no-local-files-only",
        action="store_false",
        dest="local_files_only",
        help="Disable local files only and allow remote downloads",
    )
    parser.add_argument(
        "--tagset",
        type=str,
        default=None,
        help="Comma-separated list of BIO tags (e.g. 'O,B-LOC,I-LOC,B-OBJ,I-OBJ,B-TIME,I-TIME')",
    )
    parser.add_argument("--no-audit-splits", action="store_true", default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument(
        "--metric-for-best-model",
        type=str,
        default=None,
        help="Metric used to select best checkpoint (default: entity_f1)",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="JSON string of tag class weights dict (e.g. '{\"O\": 1.0, \"B-LOC\": 2.5}')",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Early stopping patience epochs",
    )
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument(
        "--rtx3060",
        action="store_true",
        help="Apply RTX 3060 12GB safe FP16 training configuration preset for 8k synthetic trajectories",
    )
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run skeleton validation without GPU")
    parser.add_argument("--validate-only", action="store_true", help="Validate config and dataset only")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.config:
        config = parse_config_file(args.config)
        overrides: dict[str, Any] = {}
        if args.model_name_or_path is not None:
            overrides["model_name_or_path"] = args.model_name_or_path
            if args.local_files_only is None and Path(args.model_name_or_path).is_dir():
                overrides["local_files_only"] = True
        dataset_path = args.dataset_path or args.corpus_path
        if dataset_path is not None:
            overrides["dataset_path"] = dataset_path
        if args.output_dir is not None:
            overrides["output_dir"] = args.output_dir
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.max_seq_length is not None:
            overrides["max_seq_length"] = args.max_seq_length
        if args.learning_rate is not None:
            overrides["learning_rate"] = args.learning_rate
        if args.batch_size is not None:
            overrides["batch_size"] = args.batch_size
        if args.gradient_accumulation_steps is not None:
            overrides["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        if args.num_epochs is not None:
            overrides["num_epochs"] = args.num_epochs
        if args.warmup_ratio is not None:
            overrides["warmup_ratio"] = args.warmup_ratio
        if args.weight_decay is not None:
            overrides["weight_decay"] = args.weight_decay
        if args.fp16 is not None:
            overrides["fp16"] = args.fp16
        if args.local_files_only is not None:
            overrides["local_files_only"] = args.local_files_only
        if args.tagset is not None:
            overrides["tagset"] = tuple(t.strip() for t in args.tagset.split(",") if t.strip())
        if args.no_audit_splits:
            overrides["audit_splits"] = False
        if args.dataloader_num_workers is not None:
            overrides["dataloader_num_workers"] = args.dataloader_num_workers
        if args.save_total_limit is not None:
            overrides["save_total_limit"] = args.save_total_limit
        if args.version is not None:
            overrides["version"] = args.version
        if args.metric_for_best_model is not None:
            overrides["metric_for_best_model"] = args.metric_for_best_model
        if args.class_weights is not None:
            try:
                overrides["class_weights"] = json.loads(args.class_weights)
            except Exception:
                pass
        if args.early_stopping_patience is not None:
            overrides["early_stopping_patience"] = args.early_stopping_patience
        if args.rtx3060:
            if args.batch_size is None:
                overrides["batch_size"] = 16
            if args.gradient_accumulation_steps is None:
                overrides["gradient_accumulation_steps"] = 2
            if args.fp16 is None:
                overrides["fp16"] = True
            if args.learning_rate is None:
                overrides["learning_rate"] = 3e-5
            if args.max_seq_length is None:
                overrides["max_seq_length"] = 448
        if overrides:
            config = config.model_copy(update=overrides)
    else:
        tagset = DEFAULT_TAGSET
        if args.tagset:
            tagset = tuple(t.strip() for t in args.tagset.split(",") if t.strip())

        config_kwargs: dict[str, Any] = {
            "tagset": tagset,
        }
        if args.rtx3060:
            config_kwargs["batch_size"] = 16
            config_kwargs["gradient_accumulation_steps"] = 2
            config_kwargs["fp16"] = True
            config_kwargs["learning_rate"] = 3e-5
            config_kwargs["max_seq_length"] = 448
            config_kwargs["dataloader_num_workers"] = 2
            config_kwargs["save_total_limit"] = 2
        if args.model_name_or_path is not None:
            config_kwargs["model_name_or_path"] = args.model_name_or_path
        dataset_path = args.dataset_path or args.corpus_path
        if dataset_path is not None:
            config_kwargs["dataset_path"] = dataset_path
        if args.output_dir is not None:
            config_kwargs["output_dir"] = args.output_dir
        if args.seed is not None:
            config_kwargs["seed"] = args.seed
        if args.max_seq_length is not None:
            config_kwargs["max_seq_length"] = args.max_seq_length
        if args.learning_rate is not None:
            config_kwargs["learning_rate"] = args.learning_rate
        if args.batch_size is not None:
            config_kwargs["batch_size"] = args.batch_size
        if args.gradient_accumulation_steps is not None:
            config_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        if args.num_epochs is not None:
            config_kwargs["num_epochs"] = args.num_epochs
        if args.warmup_ratio is not None:
            config_kwargs["warmup_ratio"] = args.warmup_ratio
        if args.weight_decay is not None:
            config_kwargs["weight_decay"] = args.weight_decay
        if args.fp16 is not None:
            config_kwargs["fp16"] = args.fp16
        if args.local_files_only is not None:
            config_kwargs["local_files_only"] = args.local_files_only
        if args.no_audit_splits:
            config_kwargs["audit_splits"] = False
        if args.dataloader_num_workers is not None:
            config_kwargs["dataloader_num_workers"] = args.dataloader_num_workers
        if args.save_total_limit is not None:
            config_kwargs["save_total_limit"] = args.save_total_limit
        if args.version is not None:
            config_kwargs["version"] = args.version
        if args.metric_for_best_model is not None:
            config_kwargs["metric_for_best_model"] = args.metric_for_best_model
        if args.class_weights is not None:
            try:
                config_kwargs["class_weights"] = json.loads(args.class_weights)
            except Exception:
                pass
        if args.early_stopping_patience is not None:
            config_kwargs["early_stopping_patience"] = args.early_stopping_patience
        config = NERTrainConfig(**config_kwargs)

    try:
        run_train_ner(config, dry_run=args.dry_run, validate_only=args.validate_only)
        return 0
    except OptionalDependencyError as e:
        sys.stderr.write(f"Optional Dependency Error: {e}\n")
        return 1
    except DatasetAuditError as e:
        sys.stderr.write(f"Dataset Split Audit Error: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"NER Training Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
