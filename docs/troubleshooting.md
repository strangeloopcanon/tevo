# Troubleshooting

Common issues and debugging tips for running evolution experiments.

## Common Issues

### Tokenizer warnings (`TOKENIZERS_PARALLELISM`)

```bash
export TOKENIZERS_PARALLELISM=false
```

### Out of Memory (OOM)

- Reduce `batch_size` in config
- Enable `grad_ckpt: true` in train config
- Use `--device cpu` for smoke tests

### NaN losses / Instability

- Lower `instability_threshold` in train config to catch unstable models earlier
- Check that learning rate isn't too high
- Some mutations produce unstable architectures -- this is expected; they get filtered
- On CUDA, prefer bf16: set `train.bf16: true` (fp16 currently uses autocast without a GradScaler)

### MPS (Apple Silicon) quirks

- Some operations fall back to CPU; this is normal
- FlashAttention doesn't work on MPS; uses PyTorch SDPA instead
- fp16 checkpoints work well for disk savings

### CUDA out of memory

- Reduce population size
- Use gradient checkpointing
- Try smaller `max_tokens` budget

## Debugging tips

- Check `runs/<run>/live.log` for detailed output
- Use `--generations 1` to test a single generation
- The `stop_reason` field in `frontier.json` indicates why training stopped (0=normal, 1=instability, 3=early stop)
