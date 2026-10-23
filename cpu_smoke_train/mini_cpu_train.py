import json
import os
from pathlib import Path

# 外网下载（HuggingFace）走本地 7897 代理，避免 F5/无代理终端时连接被重置。
# 已缓存的模型不受影响；只有首次下载需要代理。
os.environ.setdefault("http_proxy", "http://127.0.0.1:7897")
os.environ.setdefault("https_proxy", "http://127.0.0.1:7897")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


# 使用本地模型目录（离线，不联网）
MODEL_NAME = str(Path(__file__).resolve().parent.parent / "models" / "tinyllama-1.1b-chat")
OUT_DIR = Path(__file__).resolve().parent / "output"
DATA_PATH = Path(__file__).resolve().parent / "mini_data.jsonl"


def build_data():
    """调用 expand_data.py 生成数据（单一数据源，避免硬编码样本被覆盖）"""
    import subprocess, sys
    expand_script = Path(__file__).resolve().parent / "expand_data.py"
    if expand_script.exists():
        subprocess.run([sys.executable, str(expand_script)], check=True, capture_output=True, text=True)
        print(f"[INFO] 通过 expand_data.py 生成数据: {DATA_PATH}")
    else:
        # 回退：expand_data.py 不存在时用内置最小数据
        items = [
            {"instruction": "问候", "input": "", "output": "你好，我是小助手。"},
            {"instruction": "分类", "input": "苹果", "output": "水果"},
        ]
        with DATA_PATH.open("w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[INFO] expand_data.py 不存在，使用内置 2 条样本: {DATA_PATH}")


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
    result = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    result["labels"] = result["input_ids"].copy()
    return result


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
    # 划分训练集 / 验证集：80% 训练，20% 验证（防过拟合）
    split = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"[INFO] 训练集: {len(train_ds)} 条, 验证集: {len(eval_ds)} 条")

    train_tokenized = train_ds.map(lambda batch: tokenize(batch, tokenizer), batched=True)
    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_tokenized = eval_ds.map(lambda batch: tokenize(batch, tokenizer), batched=True)
    eval_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        num_train_epochs=8,
        logging_steps=20,
        eval_strategy="steps",       # 每个 step 算一次验证集 loss
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
        disable_tqdm=False,
        load_best_model_at_end=False,
    )

    collator = lambda features: {k: torch.stack([torch.tensor(f[k]) for f in features]) for k in ["input_ids", "attention_mask", "labels"]}
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,   # 验证集
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(OUT_DIR / "final_lora"))
    print("[INFO] training finished, model saved to:", OUT_DIR / "final_lora")


if __name__ == "__main__":
    main()
