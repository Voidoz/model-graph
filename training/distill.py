"""Offline distillation recipe driver (T-0025).

STATUS: UNVERIFIED IN THIS REPOSITORY'S DEVELOPMENT ENVIRONMENT. The runnable distillation
rehearsal is `mg train specialist --config training/configs/distill-synthetic.yaml`; it uses
recorded dataset outputs as teacher samples and never calls a teacher.

This script documents the real pipeline:
  1. sample a licensed teacher on the domain slice (network/live provider),
  2. filter with deterministic tests, then a judge or human review,
  3. keep hard/failure cases and record every filter decision,
  4. train the student with the LoRA recipe in training/sft_lora.py,
  5. evaluate the served student and submit the passing report to the registry.

`--sample` and `--train` fail closed when the teacher endpoint or real training stack is
absent. They never manufacture a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import sft_lora  # noqa: E402
from modelgraph.training.specialist import (  # noqa: E402
    SpecialistConfig,
    load_specialist_config,
    load_specialist_dataset,
    resolve_path,
)


def plan(config: SpecialistConfig) -> dict[str, object]:
    distill = config.distillation
    assert distill is not None
    return {
        "config": config.source_path,
        "config_hash": config.config_hash(),
        "teacher": distill.teacher,
        "teacher_license": distill.teacher_license,
        "teacher_uri": distill.teacher_uri,
        "filter": distill.filter,
        "min_bytes": distill.min_bytes,
        "include_failures": distill.include_failures,
        "student": config.model.base,
        "hyperparameters": config.hyperparameters.model_dump(mode="json", exclude_none=True),
        "hardware": config.hardware.model_dump(mode="json", exclude_none=True),
        "output": str(resolve_path(config.output_dir)),
        "seed": config.seed,
        "steps_implemented_offline": ["recorded-output filtering", "student training recipe"],
        "steps_requiring_live_environment": ["teacher sampling", "judge review"],
        "note": (
            "Teacher sampling requires a live provider and a license decision; it is not "
            "implemented in this offline kit. The rehearsal never invents teacher samples."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample", action="store_true", help="Teacher sampling (needs network).")
    parser.add_argument("--train", action="store_true", help="Student training (needs real stack).")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_specialist_config(args.config)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2
    if config.method != "distill":
        print("this script requires method: distill", file=sys.stderr)
        return 2
    document = plan(config)
    print(json.dumps(document, indent=2, sort_keys=True))
    if not args.json:
        print(
            "\nRehearsal (offline, no teacher calls):\n"
            "  uv run mg train specialist --config training/configs/distill-synthetic.yaml --eval"
        )
        print(
            "\nReal student training (after teacher sampling and filtering):\n"
            "  uv run python training/sft_lora.py "
            f"--config {args.config} --execute"
        )
    if args.sample:
        print(
            "teacher sampling is not implemented in the offline kit: it requires a live, "
            "licensed teacher provider. Use the recorded-output rehearsal or add a provider "
            "integration with its own manifest and network gate.",
            file=sys.stderr,
        )
        return 3
    if args.train:
        dataset_path = resolve_path(config.dataset)
        if not dataset_path.exists():
            print(f"dataset not found: {dataset_path}", file=sys.stderr)
            return 2
        dataset = load_specialist_dataset(
            dataset_path,
            node=config.node,
            distillation=config.distillation,
            seed=config.seed,
        )
        missing = sft_lora.missing_modules(config)
        if missing:
            print(
                "real training stack is not installed (missing: "
                + ", ".join(missing)
                + "). Refusing to fabricate a distilled student.",
                file=sys.stderr,
            )
            return 3
        try:
            record = sft_lora._execute_training(config, dataset)  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"student training refused: {exc}", file=sys.stderr)
            return 3
        print(f"adapter: {record['adapter']}")
        print("next: evaluate the served student and submit the passing report to the registry.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
