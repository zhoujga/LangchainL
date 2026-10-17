from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "qwen3_lora_train_config.yaml"


def normalize_pasted_yaml(text: str) -> str:
    """Repair common formatting issues from pasted YAML snippets."""
    text = text.replace("Qwen/Qwen3-0.6Btrust_remote_code: true", "Qwen/Qwen3-0.6B\ntrust_remote_code: true")
    text = text.replace("Qwen/Qwen3-0.6Btrust_remote_code: true", "Qwen/Qwen3-0.6B\ntrust_remote_code: true")
    text = re.sub(r"(?P<path>/mnt/data/.+?/Qwen/Qwen3-0\.6B)\s*trust_remote_code\s*:\s*true", r"\g<path>\ntrust_remote_code: true", text)
    text = re.sub(r"(?P<path>/mnt/data/.+?/Qwen/Qwen3-0\.6B)\s*template\s*:", r"\g<path>\ntemplate:", text)
    return text


def load_config(path: Path):
    raw_text = path.read_text(encoding="utf-8")
    cleaned_text = normalize_pasted_yaml(raw_text)
    if cleaned_text != raw_text:
        print("[INFO] Repaired common YAML formatting issue in the pasted config.")
    try:
        return yaml.safe_load(cleaned_text)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] YAML parse failure: {exc}")
        raise


def validate_config(cfg: dict):
    errors: list[str] = []
    warnings: list[str] = []

    required_keys = [
        "model_name_or_path",
        "template",
        "stage",
        "finetuning_type",
        "dataset",
        "dataset_dir",
        "cutoff_len",
        "do_train",
        "num_train_epochs",
        "learning_rate",
        "lora_rank",
        "lora_alpha",
        "bf16",
        "output_dir",
    ]

    for key in required_keys:
        if cfg.get(key) is None:
            errors.append(f"missing required key: {key}")

    if cfg.get("model_name_or_path") and not str(cfg["model_name_or_path"]).startswith("/"):
        warnings.append("model_name_or_path looks non-absolute; check the environment path.")

    if cfg.get("trust_remote_code") is True and cfg.get("model_name_or_path", "").endswith("Qwen3-0.6B"):
        print("[OK] trust_remote_code is appropriate for Qwen models that require custom code.")

    lora_rank = cfg.get("lora_rank")
    lora_alpha = cfg.get("lora_alpha")
    if isinstance(lora_rank, int) and isinstance(lora_alpha, int) and lora_rank > 0 and lora_alpha >= lora_rank:
        print(f"[OK] LoRA scale looks healthy: rank={lora_rank}, alpha={lora_alpha}.")
    elif isinstance(lora_rank, int) and isinstance(lora_alpha, int):
        warnings.append("LoRA alpha is smaller than rank; this is sometimes valid but check the intended scaling.")

    if cfg.get("load_best_model_at_end") and cfg.get("save_strategy") == "steps":
        print("[OK] Best-model checkpointing is enabled with step-level saving.")

    if cfg.get("val_size") and cfg.get("eval_strategy") == "steps":
        print("[OK] Validation split + best checkpoint selection is a sensible setup for early stopping on loss.")
    else:
        warnings.append("A validation split is recommended when using load_best_model_at_end.")

    if cfg.get("warmup_steps", 0) == 0 and cfg.get("lr_scheduler_type") == "cosine":
        warnings.append("warmup_steps=0 with cosine scheduler can produce a sharp early optimization curve; consider a small warmup (for example 5-10% of steps).")

    if cfg.get("bf16") is True:
        print("[OK] BF16 is enabled, which is usually preferred on Ampere-level GPUs or newer.")

    if cfg.get("ddp_timeout") and cfg.get("ddp_timeout", 0) > 180000000:
        warnings.append("ddp_timeout is unusually large; this is rarely needed unless the dataset is extremely slow to shard.")

    if cfg.get("pack") is False:
        pass

    if cfg.get("save_total_limit") and cfg.get("save_total_limit") < 3:
        warnings.append("save_total_limit is low; this may remove useful checkpoints before manual review.")

    if errors:
        print("\n[ERRORS]")
        for item in errors:
            print(f" - {item}")
    if warnings:
        print("\n[WARNINGS]")
        for item in warnings:
            print(f" - {item}")

    return errors, warnings


def resolve_training_command() -> list[str] | None:
    if shutil.which("llamafactory-cli"):
        return ["llamafactory-cli", "train", str(CONFIG_PATH)]

    py = sys.executable
    completion = subprocess.run(
        [py, "-m", "llamafactory", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completion.returncode == 0:
        return [py, "-m", "llamafactory", "train", str(CONFIG_PATH)]

    return None


def main():
    parser = argparse.ArgumentParser(description="Validate and optionally launch a Qwen3 LoRA training config.")
    parser.add_argument("--run", action="store_true", help="Attempt to launch the training process if the environment supports it.")
    args = parser.parse_args()

    print("[INFO] Loading config from:", CONFIG_PATH)
    cfg = load_config(CONFIG_PATH)
    print("[INFO] Parsed YAML keys:", sorted(cfg.keys()) if isinstance(cfg, dict) else type(cfg).__name__)

    errors, warnings = validate_config(cfg)

    if errors:
        print("\n[RESULT] The config is not ready to launch because required keys are missing.")
        raise SystemExit(2)

    print("\n[RESULT] The configuration is structurally reasonable for Qwen3-0.6B LoRA tuning.")
    if warnings:
        print("[RESULT] It is usable, but I recommend adjusting the warnings above before a long training run.")

    if not args.run:
        print("\n[INFO] Dry run completed. Re-run with --run to attempt actual training.")
        return

    cmd = resolve_training_command()
    if not cmd:
        print("\n[INFO] No LLaMA-Factory training entry point was found in this environment, so training was not launched.")
        return

    print(f"\n[INFO] Launching training command: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
