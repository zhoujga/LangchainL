# CPU smoke test for LoRA training

This folder is meant to verify that the training pipeline works on CPU with the smallest possible footprint.

## Run

```bash
cd d:/work/tmp/LangchainCode/cpu_smoke_train
python mini_cpu_train.py
```

## Notes

- It uses a tiny model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- It uses a tiny synthetic dataset
- It trains for 1 epoch only
- It is intentionally small so it can run on CPU without a long wait

If this works, the environment is valid for later scaling up to a larger model or GPU setup.
