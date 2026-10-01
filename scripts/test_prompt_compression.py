"""Measures exact prefill latency difference between verbose and compact prompts on GTX 1050 Ti."""

import time
import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.litert_explore.engine import LiteRtModelRunner

def main():
    r = LiteRtModelRunner(backend="gpu", gpu_target="1050ti", max_num_tokens=512)
    r.generate("warmup")

    verbose = (
        "You are the Farmer in Kaggriculture. Decide actions for this turn.\n"
        "STATUS: Day 1, Hour 2 (Step 3) | Cash: $3000 | Pos: (4,4) on tile INSIDE SHED at (4,4) - Cannot plant here. Move NORTH/WEST to farm field.\n"
        "Seeds: {'WHEAT': 0, 'CARROT': 0, 'TOMATO': 0, 'STRAWBERRY': 0, 'MELON': 0} (Total: 0).\n"
        "Carried Crops: {} (Total: 0).\n"
        "Shed Inventory: {'WHEAT': 0, 'CARROT': 0, 'TOMATO': 0, 'STRAWBERRY': 0, 'MELON': 0, 'EGG': 0, 'MILK': 0, 'WOOL': 0, 'FERTILIZER': 0, 'GOOSE': 0, 'COW': 0, 'SHEEP': 0}.\n"
        "Market Prices: Wheat=$25, Carrot=$35, Melon=$250.\n"
        "IMMEDIATE TACTICAL DIRECTIVES:\n"
        "- NO SEEDS: Must add market order [['BUY_SEED', 'WHEAT', 5]].\n"
        "- In shed: farmer action should be ['NORTH'] to walk to the field.\n"
        "Reply ONLY with JSON: {'farmer': ['ACTION', ...], 'market': [['ORDER', ...]]}\n"
        "Action JSON:"
    )

    compact = (
        "KAGGRICULTURE: Turn D1:H2:S3 | Cash: $3000 | Pos: (4,4) SHED | Seeds: 0 | Carried: 0\n"
        "Rules: If seeds=0, BUY_SEED WHEAT 5. If in shed, NORTH.\n"
        "Reply JSON: {\"farmer\": [\"NORTH\"], \"market\": [[\"BUY_SEED\", \"WHEAT\", 5]]}"
    )

    t0 = time.perf_counter()
    res_v = r.generate(verbose)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    res_c = r.generate(compact)
    t3 = time.perf_counter()

    lat_verbose = t1 - t0
    lat_compact = t3 - t2

    print("=" * 60)
    print("COMPARATIVO DE LATENCIA POR DENSIDADE DE TOKEN (GTX 1050 Ti):")
    print("=" * 60)
    print(f"Prompt Verboso (~250 tokens): {lat_verbose:.3f} s")
    print(f"Prompt Compacto (~45 tokens) : {lat_compact:.3f} s")
    print(f"Speedup de Prefill/Decode   : {lat_verbose / max(lat_compact, 0.001):.2f}x mais rapido!")
    print("=" * 60)

if __name__ == "__main__":
    main()
