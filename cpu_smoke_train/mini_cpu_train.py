import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT_DIR = Path(__file__).resolve().parent / "output"
DATA_PATH = Path(__file__).resolve().parent / "mini_data.jsonl"


def build_data():
    items = [
        {"instruction": "问候", "input": "", "output": "你好，我是小助手。"},
        {"instruction": "总结", "input": "今天天气不错。", "output": "今天的天气不错。"},
        {"instruction": "翻译", "input": "hello", "output": "你好"},
        {"instruction": "分类", "input": "苹果", "output": "水果"},
    ]
    with DATA_PATH.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[INFO] created mini dataset: {DATA_PATH}")


def load_dataset():
    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = f"Instruction: {obj['instruction']}\nInput: {obj['input']}\nOutput: {obj['output']}"
            rows.append({"text": prompt})
    return Dataset.from_list(rows)


def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)


def main():
    print("[INFO] device:", "cuda" if torch.cuda.is_available() else "cpu")
    build_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model = model.to("cpu")

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    ds = load_dataset()
    tokenized = ds.map(lambda batch: tokenize(batch, tokenizer), batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=20,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        no_cuda=not torch.cuda.is_available(),
        dataloader_num_workers=0,
        disable_tqdm=False,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=lambda features: {k: torch.stack([torch.tensor(f[k]) for f in features]) for k in ["input_ids", "attention_mask", "labels"]},
    )

    trainer.train()
    trainer.save_model(str(OUT_DIR / "final_lora"))
    print("[INFO] training finished, model saved to:", OUT_DIR / "final_lora")


if __name__ == "__main__":
    main()
