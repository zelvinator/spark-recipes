#!/usr/bin/env python3
# fix-kimi-tokenizer — MANDATORY for Kimi K2.5 / K2.6 on TF5-era vLLM images.
#
# K2.5's tokenization_kimi.py imports bytes_to_unicode from
#   transformers.models.gpt2.tokenization_gpt2
# which was moved to transformers.convert_slow_tokenizer in recent transformers.
# Without this patch, model loading fails on `from transformers... import ...`.
#
# Source: OllieJW, NVIDIA forum t/362958 post #11.
# Applies to BOTH the HF hub snapshot copy and the transformers_modules cache
# copy that vLLM materializes from --trust-remote-code.
#
# Idempotent: skips if already patched; errors if old import survives a patch.
import os, sys, glob

OLD = "from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode"
NEW = "from transformers.convert_slow_tokenizer import bytes_to_unicode"

HF_HOME = os.environ.get("HF_HOME", "/root/.cache/huggingface")
TARGETS = []

# 1. Hub snapshot copy
snap = os.path.join(HF_HOME, "hub", "models--QuantTrio--Kimi-K2.5-E304", "snapshots")
TARGETS += glob.glob(os.path.join(snap, "*", "tokenization_kimi.py"))

# 2. transformers_modules cache (materialized at runtime by --trust-remote-code)
TARGETS += glob.glob(os.path.join(HF_HOME, "modules", "transformers_modules",
                                  "*", "tokenization_kimi.py"))

if not TARGETS:
    print("[fix-kimi-tokenizer] WARN: no tokenization_kimi.py found under",
          HF_HOME, file=sys.stderr)
    print("                    Run AFTER the model snapshot is present "
          "(first launch will download it; re-run the mod).", file=sys.stderr)
    sys.exit(0)

patched = 0
for f in TARGETS:
    if not os.path.isfile(f):
        continue
    txt = open(f).read()
    if OLD not in txt:
        # verify it's not in a broken half-patched state
        if NEW in txt:
            print(f"[fix-kimi-tokenizer] already patched: {f}")
        else:
            print(f"[fix-kimi-tokenizer] no target import in (skipped): {f}")
        continue
    txt = txt.replace(OLD, NEW)
    open(f, "w").write(txt)
    print(f"[fix-kimi-tokenizer] patched: {f}")
    patched += 1

# Verify no file still carries the old import
for f in TARGETS:
    if os.path.isfile(f) and OLD in open(f).read():
        print(f"[fix-kimi-tokenizer] ERROR: old import still present in {f}",
              file=sys.stderr)
        sys.exit(1)

print(f"[fix-kimi-tokenizer] OK patched_files={patched}")
