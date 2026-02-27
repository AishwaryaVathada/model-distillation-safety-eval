# Hardware and Deployment Notes

## Why external hardware is required

If the teacher is a 671B-class model, hosting it and generating data at scale requires external compute:
- multi-GPU (often multi-node) inference servers for acceptable throughput
- high VRAM GPUs (A100/H100 class) and fast interconnects
- large storage for datasets, checkpoints, and logs

For realistic distillation runs of 7B+ students, a dedicated GPU server is typically required.

## Practical options

1. Use an OpenAI-compatible API for the teacher (recommended for 671B teachers)
   - Configure `OPENAI_BASE_URL` and `OPENAI_API_KEY`
   - This repo uses `/v1/chat/completions` to stay provider-agnostic

2. Host a teacher on your own server (vLLM/TGI/etc.)
   - Provide an OpenAI-compatible endpoint or implement a custom teacher adapter

3. Train student models on:
   - a university cluster
   - a rented GPU instance
   - a self-hosted multi-GPU machine

## Minimum local test

You can run the smoke test on CPU with the dummy teacher.
You can run small-model LoRA training on a single consumer GPU for experimentation.
