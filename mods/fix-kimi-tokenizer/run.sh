#!/bin/bash
# fix-kimi-tokenizer — MANDATORY for Kimi K2.5 / K2.6 on TF5-era vLLM images.
#
# Patches the bytes_to_unicode import in tokenization_kimi.py in both the
# HF hub snapshot and the transformers_modules cache. See apply_fix.py.
#
# Fails the launch if the old import survives a patch (hard error);
# no-ops with a warning if the model snapshot is not present yet.
set -e
exec python3 "$(dirname "$0")/apply_fix.py"
