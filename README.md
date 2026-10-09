# spark-recipes

Custom mods and recipes for [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — DGX Spark vLLM builds.

This is a personal collection, structured as a drop-in overlay for a clone of upstream: copy `mods/` and `recipes/` over your `spark-vllm-docker` checkout, then build and run as usual (see the upstream README for `build-and-copy.sh` / `run-recipe.sh` usage).

Based on upstream commit `73f01ce` ("Added DSV4.1F recipes").

## mods/

Custom build-time mods:

- `diag-glm5next-b12x-attention`
- `diag-glm5next-first-nan`
- `diag-glm5next-no-b12x-kda-decode`
- `dsv41-boot10-patches`
- `fix-b12x-fp8-linear-fallback`
- `fix-glm5next-b12x-sparse-prefill-compaction`
- `fix-glm5next-bf16-layers`
- `fix-glm5next-nvfp4-ignore`
- `fix-kimi-tokenizer`
- `fix-minimax-m3-compressed-tensors`
- `fix-minimax-m3-reasoning-45718`
- `fix-minimax-m3-tool-parser`
- `fix-nccl-2.30.4`
- `glm53-flash-indexer`

## recipes/

Custom launch recipes:

- `recipes/deepseek-v4-1-flash.yaml`
- `recipes/deepseek-v4-flash-dspark-fast.yaml`
- `recipes/deepseek-v4-flash-dspark.yaml`
- `recipes/glm-4-5-fp8.yaml`
- `recipes/glm-4-6-fp8.yaml`
- `recipes/glm-4-7.yaml`
- `recipes/glm-5-1.yaml`
- `recipes/glm-52.yaml`
- `recipes/glm-5.3-flash-my.yaml`
- `recipes/glm-5-3-flash-sm121v8.yaml`
- `recipes/glm-5-3-flash-unc-nvfp4.yaml`
- `recipes/glm-5-3-flash.yaml`
- `recipes/glm-5-3-int4int8.yaml`
- `recipes/mimo-v25.yaml`
- `recipes/mimo.yaml`
- `recipes/minimax-m3.yaml`
- `recipes/nemotron-3-ultra-nvfp4.yaml`
- `recipes/nemotron-no-ray.yaml`
- `recipes/qwen38-flash.yaml`
- `recipes/4x-spark-cluster/kimi-k2.5-e304.yaml`

## patches/upstream-local-fixes/

Local modifications to upstream files, as diffs against upstream at the base commit:

- `0001-flashinfer-b12x-swigluoai-pin-compat-guard.patch` — `docker/patch_vllm_flashinfer_b12x_swigluoai.py` (pin-compat guard for pre-#47392 refs)
- `0002-dsv41-flash-nccl-tuning.patch` — `recipes/4x-spark-cluster/deepseek-v4.1-flash.yaml` (NCCL env tuning for 4x Spark ring)

## Attribution

Derived from [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (MIT, © Eugene Rakhmatulin).
