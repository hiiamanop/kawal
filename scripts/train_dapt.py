from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts import (
    OptionalDependencyError,
    create_artifact_item,
    create_or_update_artifact_manifest,
    require_ml_dependencies,
    set_deterministic_seed,
)
from services.dataset.generator import DEFAULT_TEMPLATES, generate_dataset
from services.ml.manifest import ArtifactManifest, compute_file_sha256

NIK_REGEX = re.compile(r"\b\d{16}\b")
PHONE_REGEX = re.compile(r"\b(?:\+62|62|08)[0-9]{8,12}\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def _supports_kwarg(fn_or_cls: Any, kwarg_name: str) -> bool:
    try:
        import inspect

        target = fn_or_cls.__init__ if isinstance(fn_or_cls, type) else fn_or_cls
        sig = inspect.signature(target)
        return kwarg_name in sig.parameters
    except (ValueError, TypeError, AttributeError):
        return False


class DAPTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name_or_path: str = Field(default="indobenchmark/indobert-base-p1", min_length=1)
    corpus_path: str | None = None
    output_dir: str = Field(default="artifacts/dapt", min_length=1)
    seed: int = Field(default=42, ge=0)
    max_seq_length: int = Field(default=448, ge=16, le=512)
    mlm_probability: float = Field(default=0.15, gt=0.0, lt=1.0)
    learning_rate: float = Field(default=2e-5, gt=0.0)
    batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    num_epochs: int = Field(default=3, ge=1)
    warmup_ratio: float = Field(default=0.06, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    fp16: bool = True
    local_files_only: bool = False
    dataloader_num_workers: int = Field(default=2, ge=0)
    save_total_limit: int = Field(default=2, ge=1)
    minhash_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    pii_scrubbing: bool = True
    version: str = Field(default="v1.0.0", min_length=1)

    @classmethod
    def for_rtx3060(cls, corpus_path: str | None = None, **kwargs: Any) -> DAPTConfig:
        """Create DAPT training configuration with safe RTX 3060 12GB FP16 defaults for ~8k trajectories."""
        defaults: dict[str, Any] = {
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "learning_rate": 2e-5,
            "max_seq_length": 448,
            "dataloader_num_workers": 2,
            "save_total_limit": 2,
        }
        defaults.update(kwargs)
        return cls(corpus_path=corpus_path, **defaults)


def get_rtx3060_dapt_config(corpus_path: str | None = None, **kwargs: Any) -> DAPTConfig:
    """Convenience helper returning RTX 3060 safe FP16 DAPT training config."""
    return DAPTConfig.for_rtx3060(corpus_path=corpus_path, **kwargs)


def parse_dapt_config_dict(data: dict[str, Any]) -> DAPTConfig:
    """Parse DAPTConfig from either flat JSON or bundled M3TrainingConfig schema."""
    if "hyperparameters" in data and "model_version" in data:
        hp = data.get("hyperparameters", {})
        mv = data.get("model_version", {})
        extra = hp.get("extra_params", {})
        prec = data.get("precision_config", {})
        return DAPTConfig(
            model_name_or_path=mv.get("base_model", "indobenchmark/indobert-base-p1"),
            corpus_path=extra.get("corpus_path") or data.get("corpus_path"),
            output_dir=extra.get("output_dir", "artifacts/dapt"),
            seed=data.get("seed", 42),
            max_seq_length=mv.get("max_sequence_length", 448),
            mlm_probability=hp.get("masking_probability", 0.15),
            learning_rate=hp.get("learning_rate", 2e-5),
            batch_size=hp.get("batch_size", 8),
            gradient_accumulation_steps=hp.get("gradient_accumulation_steps", 4),
            num_epochs=hp.get("num_train_epochs", 3),
            warmup_ratio=hp.get("warmup_ratio", 0.06),
            weight_decay=hp.get("weight_decay", 0.01),
            fp16=extra.get("fp16", prec.get("training_precision") == "fp16"),
            local_files_only=extra.get("local_files_only", False),
            dataloader_num_workers=extra.get("dataloader_num_workers", 2),
            save_total_limit=extra.get("save_total_limit", 2),
            minhash_threshold=extra.get("minhash_threshold", 0.85),
            pii_scrubbing=extra.get("pii_scrubbing", True),
            version=mv.get("model_version", "v1.0.0"),
        )
    return DAPTConfig.model_validate(data)


def parse_config_file(config_path: str | Path) -> DAPTConfig:
    """Load and parse DAPTConfig from file."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return parse_dapt_config_dict(data)


def validate_corpus_text(
    lines: list[str],
    pii_scrubbing: bool = True,
    minhash_threshold: float = 0.85,
) -> list[str]:
    """Validate and clean corpus lines, verifying PII scrubbing and deduplication."""
    cleaned_lines: list[str] = []
    seen_ngrams: list[set[tuple[str, ...]]] = []

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        if pii_scrubbing:
            if NIK_REGEX.search(stripped):
                raise ValueError(f"PII violation (NIK detected) in corpus line {idx}")
            if PHONE_REGEX.search(stripped):
                raise ValueError(f"PII violation (Phone number detected) in corpus line {idx}")
            if EMAIL_REGEX.search(stripped):
                raise ValueError(f"PII violation (Email detected) in corpus line {idx}")

        words = stripped.lower().split()
        if len(words) >= 3:
            grams = {tuple(words[i : i + 3]) for i in range(len(words) - 2)}
            is_dup = False
            for existing in seen_ngrams:
                intersection = len(grams & existing)
                union = len(grams | existing)
                if union > 0 and (intersection / union) >= minhash_threshold:
                    is_dup = True
                    break
            if not is_dup:
                seen_ngrams.append(grams)
                cleaned_lines.append(stripped)
        else:
            cleaned_lines.append(stripped)

    return cleaned_lines


def _is_train_provenance(data: Any) -> bool:
    """Ensure data provenance belongs strictly to TRAIN split, filtering out dev/test."""
    if isinstance(data, str):
        return True
    if isinstance(data, dict):
        split = data.get("split")
        if split is not None:
            split_str = str(split.value if hasattr(split, "value") else split).strip().lower()
            return split_str == "train"
        wt = data.get("world_truth")
        if isinstance(wt, dict) and "split" in wt:
            return str(wt["split"]).strip().lower() == "train"
        return True
    if hasattr(data, "split") and not callable(getattr(data, "split")):
        split_val = getattr(data, "split")
        split_str = str(split_val.value if hasattr(split_val, "value") else split_val).strip().lower()
        return split_str == "train"
    return True


def _extract_text_from_json_item(data: Any, raw_lines: list[str], fallback_str: str = "") -> None:
    if isinstance(data, dict) and not _is_train_provenance(data):
        return
    if isinstance(data, str):
        if data.strip():
            raw_lines.append(data.strip())
        return
    if not isinstance(data, dict):
        if fallback_str.strip():
            raw_lines.append(fallback_str.strip())
        return

    if "turns" in data and isinstance(data["turns"], (list, tuple)):
        extracted_any = False
        for turn in data["turns"]:
            if isinstance(turn, dict):
                bubbles = turn.get("bubbles", [])
                if isinstance(bubbles, (list, tuple)):
                    for bubble in bubbles:
                        if isinstance(bubble, dict) and bubble.get("text"):
                            raw_lines.append(str(bubble["text"]).strip())
                            extracted_any = True
                        elif hasattr(bubble, "text") and bubble.text:
                            raw_lines.append(str(bubble.text).strip())
                            extracted_any = True
            elif hasattr(turn, "bubbles"):
                for bubble in getattr(turn, "bubbles", []):
                    if hasattr(bubble, "text") and bubble.text:
                        raw_lines.append(str(bubble.text).strip())
                        extracted_any = True
        if extracted_any:
            return

    for key in ("text", "content", "line", "body", "message", "sentence", "complaint"):
        if key in data and isinstance(data[key], str) and data[key].strip():
            raw_lines.append(data[key].strip())
            return

    if fallback_str.strip():
        raw_lines.append(fallback_str.strip())


def load_corpus(
    corpus_path: str | Path | None,
    pii_scrubbing: bool = True,
    minhash_threshold: float = 0.85,
    seed: int = 42,
) -> list[str]:
    """Load corpus from file (plain text, JSON array, or JSONL) or generate domain corpus from synthetic generator."""
    raw_lines: list[str] = []
    if corpus_path is not None:
        p = Path(corpus_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Corpus file not found: {p}")
        if p.suffix.lower() == ".json":
            with p.open("r", encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("["):
                try:
                    data_list = json.loads(content)
                    if isinstance(data_list, list):
                        for item in data_list:
                            if _is_train_provenance(item):
                                _extract_text_from_json_item(item, raw_lines)
                except json.JSONDecodeError:
                    pass
            elif content.startswith("{"):
                try:
                    data_dict = json.loads(content)
                    if isinstance(data_dict, dict) and _is_train_provenance(data_dict):
                        _extract_text_from_json_item(data_dict, raw_lines, content)
                except json.JSONDecodeError:
                    pass
            if not raw_lines:
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped:
                        raw_lines.append(stripped)
        else:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if p.suffix.lower() == ".jsonl" or (stripped.startswith("{") and stripped.endswith("}")):
                        try:
                            data = json.loads(stripped)
                            if not _is_train_provenance(data):
                                continue
                            _extract_text_from_json_item(data, raw_lines, stripped)
                        except json.JSONDecodeError:
                            raw_lines.append(stripped)
                    else:
                        raw_lines.append(stripped)
    else:
        raw_lines = [
            f"Laporan masyarakat: {tmpl.issue} di lokasi {tmpl.landmark}, {tmpl.jurisdiction}, {tmpl.city}."
            for tmpl in DEFAULT_TEMPLATES
        ]
        synthetic_trajectories = generate_dataset(num_scenarios=36, seed=seed)
        train_trajectories = [t for t in synthetic_trajectories if _is_train_provenance(t)]
        for traj in train_trajectories:
            for turn in traj.turns:
                for bubble in turn.bubbles:
                    if bubble.text and bubble.text.strip():
                        raw_lines.append(bubble.text.strip())

    return validate_corpus_text(
        raw_lines,
        pii_scrubbing=pii_scrubbing,
        minhash_threshold=minhash_threshold,
    )


def compute_dataset_lineage(
    corpus_path: str | Path | None,
    corpus: list[str] | None = None,
    repo_root_dir: Path | None = None,
) -> dict[str, Any]:
    if repo_root_dir is None:
        repo_root_dir = repo_root

    line_count = len(corpus) if corpus is not None else 0
    cleaned_record_count = line_count

    if corpus_path is None:
        synthetic_hash = None
        if corpus:
            synthetic_hash = hashlib.sha256("\n".join(corpus).encode("utf-8")).hexdigest().lower()

        return {
            "source": "synthetic_generator",
            "corpus_path": None,
            "corpus_relative_path": None,
            "corpus_basename": None,
            "corpus_sha256": synthetic_hash,
            "sha256": synthetic_hash,
            "raw_record_count": line_count,
            "record_count": line_count,
            "line_count": line_count,
            "cleaned_record_count": cleaned_record_count,
            "is_synthetic": True,
        }

    p = Path(corpus_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Corpus file not found: {p}")

    basename = p.name
    rel_path = basename
    raw_given = Path(corpus_path)
    if not raw_given.is_absolute() and ".." not in raw_given.parts:
        rel_path = raw_given.as_posix()
    else:
        try:
            rel_path = p.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            try:
                rel_path = p.relative_to(repo_root_dir.resolve()).as_posix()
            except ValueError:
                rel_path = basename

    file_sha256 = compute_file_sha256(p)
    raw_record_count = 0
    raw_lines = 0

    if p.suffix.lower() == ".json":
        content = p.read_text(encoding="utf-8")
        stripped = content.strip()
        if stripped.startswith("["):
            try:
                data = json.loads(stripped)
                if isinstance(data, list):
                    raw_record_count = len(data)
            except Exception:
                raw_record_count = len([line for line in stripped.splitlines() if line.strip()])
        elif stripped.startswith("{"):
            raw_record_count = 1
        else:
            raw_record_count = len([line for line in stripped.splitlines() if line.strip()])
        raw_lines = len(content.splitlines())
    else:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                raw_lines += chunk.count(b"\n")
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    raw_record_count += 1
        if raw_lines == 0 and raw_record_count > 0:
            raw_lines = raw_record_count

    return {
        "source": "file",
        "corpus_path": rel_path,
        "corpus_relative_path": rel_path,
        "corpus_basename": basename,
        "corpus_sha256": file_sha256,
        "sha256": file_sha256,
        "raw_record_count": raw_record_count,
        "record_count": line_count if corpus is not None else raw_record_count,
        "line_count": line_count if corpus is not None else raw_lines,
        "cleaned_record_count": cleaned_record_count,
        "is_synthetic": False,
    }


def write_dapt_metadata(
    output_dir: Path | str,
    config: DAPTConfig,
    lineage: dict[str, Any],
    corpus_len: int,
) -> dict[str, Any]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_payload: dict[str, Any] = {
        "model_type": "IndoBERT-DAPT",
        "base_model": config.model_name_or_path,
        "version": config.version,
        "dataset_lineage": lineage,
        "corpus_path": lineage.get("corpus_path"),
        "corpus_relative_path": lineage.get("corpus_relative_path"),
        "corpus_basename": lineage.get("corpus_basename"),
        "corpus_sha256": lineage.get("corpus_sha256"),
        "sha256": lineage.get("sha256"),
        "record_count": lineage.get("record_count", corpus_len),
        "line_count": lineage.get("line_count", corpus_len),
        "raw_record_count": lineage.get("raw_record_count", corpus_len),
        "cleaned_record_count": lineage.get("cleaned_record_count", corpus_len),
        "training_lines": corpus_len,
        "max_seq_length": config.max_seq_length,
        "mlm_probability": config.mlm_probability,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_epochs": config.num_epochs,
        "seed": config.seed,
        "fp16": config.fp16,
    }

    dataset_manifest_payload: dict[str, Any] = {
        "manifest_type": "dapt-dataset-lineage",
        "version": config.version,
        "seed": config.seed,
        "dataset_lineage": lineage,
        "corpus_path": lineage.get("corpus_path"),
        "corpus_relative_path": lineage.get("corpus_relative_path"),
        "corpus_basename": lineage.get("corpus_basename"),
        "corpus_sha256": lineage.get("corpus_sha256"),
        "sha256": lineage.get("sha256"),
        "record_count": lineage.get("record_count", corpus_len),
        "line_count": lineage.get("line_count", corpus_len),
        "raw_record_count": lineage.get("raw_record_count", corpus_len),
        "cleaned_record_count": lineage.get("cleaned_record_count", corpus_len),
        "training_lines": corpus_len,
    }

    with open(out_dir / "dapt_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    with open(out_dir / "kawal_dapt_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    with open(out_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dataset_manifest_payload, f, indent=2)

    with open(out_dir / "dapt_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dataset_manifest_payload, f, indent=2)

    config_file = out_dir / "config.json"
    if config_file.is_file():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            if isinstance(cfg_data, dict):
                cfg_data.setdefault("task_specific_params", {}).setdefault("dapt", {})["dataset_lineage"] = lineage
                cfg_data["dataset_lineage"] = lineage
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(cfg_data, f, indent=2)
        except Exception:
            pass

    return metadata_payload


def run_train_dapt(
    config: DAPTConfig,
    dry_run: bool = False,
    validate_only: bool = False,
) -> ArtifactManifest | None:
    """Execute DAPT IndoBERT training or dry-run skeleton validation."""
    set_deterministic_seed(config.seed)

    corpus = load_corpus(
        config.corpus_path,
        pii_scrubbing=config.pii_scrubbing,
        minhash_threshold=config.minhash_threshold,
        seed=config.seed,
    )
    if not corpus:
        raise ValueError("Corpus is empty after validation and filtering.")

    if validate_only:
        return None

    out_dir = Path(config.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lineage = compute_dataset_lineage(
        corpus_path=config.corpus_path,
        corpus=corpus,
        repo_root_dir=repo_root,
    )

    if not dry_run:
        require_ml_dependencies("torch", "transformers", purpose="IndoBERT DAPT training")

        import torch
        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            local_files_only=config.local_files_only,
        )

        model = AutoModelForMaskedLM.from_pretrained(
            config.model_name_or_path,
            local_files_only=config.local_files_only,
        )

        tokenized = tokenizer(
            corpus,
            padding=False,
            truncation=True,
            max_length=config.max_seq_length,
            return_special_tokens_mask=True,
        )

        class DAPTCorpusDataset(torch.utils.data.Dataset):
            def __init__(self, encodings: dict[str, list[Any]]) -> None:
                self.input_ids = encodings["input_ids"]
                self.attention_mask = encodings.get("attention_mask")
                self.special_tokens_mask = encodings.get("special_tokens_mask")

            def __len__(self) -> int:
                return len(self.input_ids)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                item: dict[str, Any] = {
                    "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long)
                }
                if self.attention_mask is not None:
                    item["attention_mask"] = torch.tensor(self.attention_mask[idx], dtype=torch.long)
                if self.special_tokens_mask is not None:
                    item["special_tokens_mask"] = torch.tensor(self.special_tokens_mask[idx], dtype=torch.long)
                return item

        train_dataset = DAPTCorpusDataset(tokenized)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=config.mlm_probability,
        )

        is_cuda = torch.cuda.is_available()
        use_fp16 = bool(config.fp16 and is_cuda)

        training_args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            fp16=use_fp16,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=config.save_total_limit,
            dataloader_num_workers=config.dataloader_num_workers if is_cuda else 0,
            dataloader_pin_memory=is_cuda,
            seed=config.seed,
            report_to="none",
        )

        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "data_collator": data_collator,
        }
        if _supports_kwarg(Trainer, "processing_class"):
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = Trainer(**trainer_kwargs)

        trainer.train()

        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

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

        weight_path = target_weights

        write_dapt_metadata(
            output_dir=out_dir,
            config=config,
            lineage=lineage,
            corpus_len=len(corpus),
        )

        items = [
            create_artifact_item(
                file_path=weight_path,
                name="indobert-dapt",
                version=config.version,
                task="dapt",
                precision="fp16" if use_fp16 else "fp32",
                relative_to=out_dir,
            )
        ]
        manifest = create_or_update_artifact_manifest(
            output_dir=out_dir,
            items=items,
            environment="cuda" if is_cuda else "local-cpu",
        )
        return manifest

    config_path = out_dir / "config.json"
    hf_config = {
        "architectures": ["BertForMaskedLM"],
        "model_type": "bert",
        "base_model": config.model_name_or_path,
        "local_files_only": config.local_files_only,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 32000,
        "max_position_embeddings": config.max_seq_length,
        "task_specific_params": {
            "dapt": {
                "base_model": config.model_name_or_path,
                "mlm_probability": config.mlm_probability,
                "learning_rate": config.learning_rate,
                "seed": config.seed,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "fp16": config.fp16,
                "dataset_lineage": lineage,
            }
        },
        "dataset_lineage": lineage,
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    weight_path = out_dir / "model.safetensors"
    skeleton_payload = (
        f"KAWAL_INDOBERT_DAPT_SKELETON_V1_SEED_{config.seed}_LEN_{len(corpus)}".encode("utf-8")
        + b"\x00" * 256
    )
    weight_path.write_bytes(skeleton_payload)

    tokenizer_config_path = out_dir / "tokenizer_config.json"
    tokenizer_meta = {
        "do_lower_case": True,
        "model_max_length": config.max_seq_length,
        "tokenizer_class": "BertTokenizer",
    }
    with tokenizer_config_path.open("w", encoding="utf-8") as f:
        json.dump(tokenizer_meta, f, indent=2)

    vocab_path = out_dir / "vocab.txt"
    vocab_path.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n", encoding="utf-8")

    write_dapt_metadata(
        output_dir=out_dir,
        config=config,
        lineage=lineage,
        corpus_len=len(corpus),
    )

    items = [
        create_artifact_item(
            file_path=weight_path,
            name="indobert-dapt",
            version=config.version,
            task="dapt",
            precision="fp16" if config.fp16 else "fp32",
            relative_to=out_dir,
        )
    ]

    manifest = create_or_update_artifact_manifest(
        output_dir=out_dir,
        items=items,
        environment="local-cpu",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IndoBERT Domain-Adaptive Pre-Training (DAPT) CLI Skeleton & Validation"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="Path to pre-trained model or cached checkpoint directory",
    )
    parser.add_argument("--corpus-path", type=str, default=None, help="Path to domain text corpus")
    parser.add_argument("--dataset-path", type=str, default=None, help="Alias for corpus-path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--mlm-probability", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--fp16", action="store_true", default=None, help="Enable fp16 mixed precision")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable fp16 mixed precision")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=None,
        help="Load local cached files only without remote downloads",
    )
    parser.add_argument(
        "--no-local-files-only",
        action="store_false",
        dest="local_files_only",
        help="Disable local files only and allow remote downloads",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument(
        "--rtx3060",
        action="store_true",
        help="Apply RTX 3060 12GB safe FP16 training configuration preset for 8k synthetic trajectories",
    )
    parser.add_argument("--minhash-threshold", type=float, default=None)
    parser.add_argument("--no-pii-scrubbing", action="store_true", default=None, help="Disable PII scrubbing check")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run skeleton validation without GPU")
    parser.add_argument("--validate-only", action="store_true", help="Validate config and corpus only")
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
        corpus_path = args.corpus_path or args.dataset_path
        if corpus_path is not None:
            overrides["corpus_path"] = corpus_path
        if args.output_dir is not None:
            overrides["output_dir"] = args.output_dir
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.max_seq_length is not None:
            overrides["max_seq_length"] = args.max_seq_length
        if args.mlm_probability is not None:
            overrides["mlm_probability"] = args.mlm_probability
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
        if args.dataloader_num_workers is not None:
            overrides["dataloader_num_workers"] = args.dataloader_num_workers
        if args.save_total_limit is not None:
            overrides["save_total_limit"] = args.save_total_limit
        if args.minhash_threshold is not None:
            overrides["minhash_threshold"] = args.minhash_threshold
        if args.no_pii_scrubbing is not None:
            overrides["pii_scrubbing"] = not args.no_pii_scrubbing
        if args.version is not None:
            overrides["version"] = args.version
        if args.rtx3060:
            if args.batch_size is None:
                overrides["batch_size"] = 8
            if args.gradient_accumulation_steps is None:
                overrides["gradient_accumulation_steps"] = 4
            if args.fp16 is None:
                overrides["fp16"] = True
            if args.learning_rate is None:
                overrides["learning_rate"] = 2e-5
            if args.max_seq_length is None:
                overrides["max_seq_length"] = 448
        if overrides:
            config = config.model_copy(update=overrides)
    else:
        config_kwargs: dict[str, Any] = {}
        if args.rtx3060:
            config_kwargs["batch_size"] = 8
            config_kwargs["gradient_accumulation_steps"] = 4
            config_kwargs["fp16"] = True
            config_kwargs["learning_rate"] = 2e-5
            config_kwargs["max_seq_length"] = 448
            config_kwargs["dataloader_num_workers"] = 2
            config_kwargs["save_total_limit"] = 2
        if args.model_name_or_path is not None:
            config_kwargs["model_name_or_path"] = args.model_name_or_path
        corpus_path = args.corpus_path or args.dataset_path
        if corpus_path is not None:
            config_kwargs["corpus_path"] = corpus_path
        if args.output_dir is not None:
            config_kwargs["output_dir"] = args.output_dir
        if args.seed is not None:
            config_kwargs["seed"] = args.seed
        if args.max_seq_length is not None:
            config_kwargs["max_seq_length"] = args.max_seq_length
        if args.mlm_probability is not None:
            config_kwargs["mlm_probability"] = args.mlm_probability
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
        elif args.model_name_or_path and Path(args.model_name_or_path).is_dir():
            config_kwargs["local_files_only"] = True
        if args.dataloader_num_workers is not None:
            config_kwargs["dataloader_num_workers"] = args.dataloader_num_workers
        if args.save_total_limit is not None:
            config_kwargs["save_total_limit"] = args.save_total_limit
        if args.minhash_threshold is not None:
            config_kwargs["minhash_threshold"] = args.minhash_threshold
        if args.no_pii_scrubbing is not None:
            config_kwargs["pii_scrubbing"] = not args.no_pii_scrubbing
        if args.version is not None:
            config_kwargs["version"] = args.version
        config = DAPTConfig(**config_kwargs)

    try:
        run_train_dapt(config, dry_run=args.dry_run, validate_only=args.validate_only)
        return 0
    except OptionalDependencyError as e:
        sys.stderr.write(f"Optional Dependency Error: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"DAPT Training Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
