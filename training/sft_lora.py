"""Real LoRA/QLoRA SFT recipe driver (T-0025).

STATUS: UNVERIFIED IN THIS REPOSITORY'S DEVELOPMENT ENVIRONMENT. The development environment
has no torch/transformers/PEFT, no CUDA driver, and no GPU. This script therefore refuses to
run until those are present; it never fabricates a training result.

The deterministic CPU rehearsal is `mg train specialist --config
training/configs/specialist-synthetic.yaml`. Use this script only on a machine that meets the
hardware assumptions in `training/configs/specialist-8gb-qlora.yaml` or
`specialist-24gb-lora.yaml`.

Usage:
    uv run python training/sft_lora.py --config training/configs/24gb-lora.yaml --dry-run
    uv run python training/sft_lora.py --config training/configs/24gb-lora.yaml --execute
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from modelgraph.training.specialist import (  # noqa: E402
    SpecialistConfig,
    SpecialistDataset,
    load_specialist_config,
    load_specialist_dataset,
    resolve_path,
)

REQUIRED_MODULES = ("torch", "transformers", "peft", "datasets", "accelerate")


def missing_modules(config: SpecialistConfig) -> list[str]:
    required = list(REQUIRED_MODULES)
    if config.method == "qlora":
        required.append("bitsandbytes")
    return [name for name in required if importlib.util.find_spec(name) is None]


def plan(config: SpecialistConfig, dataset: SpecialistDataset | None) -> dict[str, Any]:
    h = config.hyperparameters
    return {
        "config": config.source_path,
        "config_hash": config.config_hash(),
        "name": config.name,
        "method": config.method,
        "mode": config.mode,
        "base": config.model.base,
        "base_license": config.model.base_license,
        "dataset": None
        if dataset is None
        else {
            "id": dataset.dataset_id,
            "version": dataset.dataset_version,
            "hash": dataset.dataset_hash,
            "license": dataset.license_id,
            "consent": dataset.consent,
            "counts": dataset.counts,
            "sources": len(dataset.sources),
        },
        "hyperparameters": {
            "rank": h.rank,
            "alpha": h.alpha,
            "dropout": h.dropout,
            "learning_rate": h.learning_rate,
            "epochs": h.epochs,
            "batch_size": h.batch_size,
            "gradient_accumulation_steps": h.gradient_accumulation_steps,
            "effective_batch": h.batch_size * h.gradient_accumulation_steps,
            "max_seq_length": h.max_seq_length,
            "quantization": h.quantization,
            "target_modules": h.target_modules,
        },
        "hardware": config.hardware.model_dump(mode="json", exclude_none=True),
        "output_dir": str(resolve_path(config.output_dir)),
        "seed": config.seed,
        "missing_modules": missing_modules(config),
        "note": "No training is performed by --dry-run.",
    }


def _execute_training(config: SpecialistConfig, dataset: SpecialistDataset) -> dict[str, Any]:
    """Run a single-GPU PEFT SFT loop. Unverified in this environment."""
    missing = missing_modules(config)
    if missing:
        raise RuntimeError(
            "real training stack is not installed (missing: "
            + ", ".join(missing)
            + "). Refusing to fabricate a LoRA run."
        )
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device is visible. The 8GB/24GB recipes require an NVIDIA GPU with a "
            "working driver; refusing to run them on CPU."
        )
    if config.method == "qlora":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None
    tokenizer = AutoTokenizer.from_pretrained(config.model.base, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto" if quantization_config is not None else None,
    )
    if config.method == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.hyperparameters.gradient_checkpointing
        )
    lora = LoraConfig(
        r=config.hyperparameters.rank,
        lora_alpha=config.hyperparameters.alpha,
        lora_dropout=config.hyperparameters.dropout,
        target_modules=config.hyperparameters.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    rows = []
    for example in dataset.examples:
        messages = list(example.messages) + [
            {"role": "assistant", "content": example.target_text}
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:  # pragma: no cover - current tokenizers implement chat templates
            text = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=config.hyperparameters.max_seq_length,
            padding=False,
        )
        encoded["labels"] = list(encoded["input_ids"])
        rows.append(encoded)
    training_dataset = Dataset.from_list(rows)
    output_dir = resolve_path(config.output_dir)
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=config.hyperparameters.batch_size,
        gradient_accumulation_steps=config.hyperparameters.gradient_accumulation_steps,
        num_train_epochs=config.hyperparameters.epochs,
        learning_rate=config.hyperparameters.learning_rate,
        lr_scheduler_type=config.hyperparameters.scheduler,
        warmup_ratio=config.hyperparameters.warmup_ratio,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        optim=config.hyperparameters.optimizer,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=config.hyperparameters.gradient_checkpointing,
        seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=training_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    result = trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    files = {
        str(path.relative_to(adapter_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(adapter_dir.rglob("*"))
        if path.is_file()
    }
    record = {
        "schema": "modelgraph/real-training-result/v0",
        "config": config.source_path,
        "config_hash": config.config_hash(),
        "dataset": {
            "id": dataset.dataset_id,
            "version": dataset.dataset_version,
            "hash": dataset.dataset_hash,
        },
        "base": config.model.base,
        "method": config.method,
        "seed": config.seed,
        "hardware": config.hardware.model_dump(mode="json", exclude_none=True),
        "steps": int(result.global_step),
        "train_loss": float(result.training_loss),
        "adapter": str(adapter_dir),
        "adapter_files": files,
        "note": (
            "Real adapter produced by training/sft_lora.py. It is not promoted: run the eval "
            "hook against a served model and submit the passing report to the registry."
        ),
    }
    (output_dir / "training-result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _load_dataset_for_plan(config: SpecialistConfig) -> SpecialistDataset | None:
    path = resolve_path(config.dataset)
    if not path.exists():
        return None
    return load_specialist_dataset(path, node=config.node)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run real training. Fails closed when the stack or GPU is absent.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan and stop (the default when --execute is absent).",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_specialist_config(args.config)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2
    if args.dataset is not None:
        config = config.model_copy(update={"dataset": str(args.dataset)})
    if args.out is not None:
        config = config.model_copy(update={"output_dir": str(args.out)})
    if config.mode != "real":
        print(
            "this config is mode: synthetic; use `mg train specialist` for the rehearsal",
            file=sys.stderr,
        )
        return 2
    dataset = _load_dataset_for_plan(config)
    resolved = plan(config, dataset)
    if args.json:
        print(json.dumps(resolved, indent=2, sort_keys=True))
    else:
        print(f"config:      {resolved['config']}")
        print(f"config hash: {resolved['config_hash']}")
        print(f"method:      {resolved['method']}")
        print(f"base:        {resolved['base']} ({resolved['base_license']})")
        print(f"output:      {resolved['output_dir']}")
        print(f"seed:        {resolved['seed']}")
        if dataset is None:
            print(f"dataset:     NOT FOUND at {resolve_path(config.dataset)}")
        else:
            print(
                f"dataset:     {dataset.dataset_id}@{dataset.dataset_version} "
                f"({dataset.dataset_hash})"
            )
            print(f"splits:      {dataset.counts}")
            print(f"license:     {dataset.license_id}")
            print(f"consent:     {dataset.consent.get('recorded')}")
        print(f"missing:     {', '.join(resolved['missing_modules']) or 'none'}")
        print("note:        no training is performed by --dry-run")
        print()
        print("Exact execute command:")
        print(
            "uv run python training/sft_lora.py "
            f"--config {args.config} --execute"
        )
    if not args.execute:
        return 0
    if dataset is None:
        print(
            f"dataset not found: {resolve_path(config.dataset)}. Export one with "
            "`mg data export` first.",
            file=sys.stderr,
        )
        return 2
    try:
        record = _execute_training(config, dataset)
    except Exception as exc:  # noqa: BLE001 - CLI boundary, fail loudly and honestly
        print(f"training refused: {exc}", file=sys.stderr)
        return 3
    print(f"adapter: {record['adapter']}")
    print(f"steps:   {record['steps']}")
    print(f"loss:    {record['train_loss']}")
    print(
        "next: evaluate the served adapter, then submit the passing report to "
        "`mg registry add`; this command does not promote."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
