"""
Probe P1: Inspect TFLite model signatures and KV cache state buffers in gemma-4-E2B-it.litertlm.
Specifically determines:
1. Model dimension, layers, KV cache tensor names and shapes.
2. Signatures: prefill and decode signatures.
3. Whether KV cache state exists for all 35 layers or only non-KV-shared layers (0-14).
4. MTP drafter model signatures.
"""

import sys
import io
import mmap
import struct
import json

MODEL_PATH = r"C:\Users\alefita\.litert-lm\cache\huggingface\litert-community\gemma-4-E2B-it-litert-lm\gemma-4-E2B-it.litertlm"

# Section offsets from litertlm peek:
# Section 10: tf_lite_prefill_decode: [1725530112, 2543805296) (818 MB)
# Section 11: tf_lite_mtp_drafter:    [2543812608, 2588138320) (44 MB)
# Section 3:  tf_lite_per_layer_embedder: [108560384, 1393078776) (1.28 GB)

def inspect_section_tflite(f, begin, end, name):
    print(f"\n=======================================================")
    print(f" Inspecting Section: {name} [{begin}..{end}] ({ (end-begin)/(1024*1024):.2f} MB)")
    print(f"=======================================================")
    
    # Read the TFLite flatbuffer header
    f.seek(begin)
    data = f.read(min(end - begin, 1024 * 1024 * 50)) # read first 50MB or full
    
    # TFLite file identifier is at offset 4: 'TFL3'
    file_ident = data[4:8]
    print(f"  Identifier: {file_ident}")
    
    # Let's inspect strings / tensor names in the buffer
    # Search for KV cache tensor names (e.g. 'k_cache', 'v_cache', 'kv_cache', 'key_cache', 'value_cache')
    import re
    kv_names = set(re.findall(rb'(?:k_cache|v_cache|key_cache|value_cache|state|kv_)[a-zA-Z0-9_\.]*', data))
    kv_sorted = sorted([n.decode('ascii', errors='ignore') for n in kv_names])
    print(f"  Found {len(kv_sorted)} KV/state tensor name matches:")
    for n in kv_sorted[:25]:
        print(f"    - {n}")
    if len(kv_sorted) > 25:
        print(f"    ... and {len(kv_sorted)-25} more")

    # Search for layer numbers in tensor names
    layer_matches = set(re.findall(rb'layer_(\d+)|layers\.(\d+)|transformer\.h\.(\d+)|model\.layers\.(\d+)', data))
    layer_nums = sorted(set(int(m[0] or m[1] or m[2] or m[3]) for m in layer_matches if any(m)))
    if layer_nums:
        print(f"  Detected Layer Indices in tensors: min={min(layer_nums)}, max={max(layer_nums)}, count={len(layer_nums)}")
        print(f"    Layers list: {layer_nums}")
        
    # Search for signature keys
    sig_keys = set(re.findall(rb'(?:prefill|decode|serving_default|main)[a-zA-Z0-9_\.]*', data))
    sig_sorted = sorted([s.decode('ascii', errors='ignore') for s in sig_keys if len(s) < 40])
    print(f"  Signature candidates found: {sig_sorted[:15]}")

def main():
    print(f"[+] Opening model file: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        # Section 10: prefill_decode
        inspect_section_tflite(f, 1725530112, 2543805296, "tf_lite_prefill_decode")
        # Section 11: mtp_drafter
        inspect_section_tflite(f, 2543812608, 2588138320, "tf_lite_mtp_drafter")

if __name__ == "__main__":
    main()
