"""
Experimento 002 - Benchmark Comparativo: Decodificação Padrão vs Especulativa (MTP Drafter) na RTX 2060
Medição empírica de throughput (tokens/s), latência e taxa de aceitação especulativa.
"""

import time
import json
from litert_lm import Engine, interfaces

MODEL_PATH = r"C:\Users\alefita\.litert-lm\cache\huggingface\litert-community\gemma-4-E2B-it-litert-lm\gemma-4-E2B-it.litertlm"

TEST_PROMPTS = [
    {
        "name": "algorithmic_reasoning",
        "prompt": "Escreva uma funcao em Python para calcular a sequencia de Fibonacci usando programacao dinamica com memoizacao e explique brevemente a complexidade assintotica O(n)."
    },
    {
        "name": "system_design",
        "prompt": "Explique em tres passos como funciona uma fila de prioridade usando uma arvore binaria heap e quais sao as garantias de tempo para insercao e remocao."
    },
    {
        "name": "code_translation",
        "prompt": "Converta a seguinte expressao matematica para uma declaracao em linguagem C com tipos inteiros de 64 bits: z = (a * b) mod m, garantindo que nao ocorra overflow intermediario."
    }
]

def extract_text(response):
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, list):
            parts = [item["text"] for item in content if isinstance(item, dict) and "text" in item]
            return "".join(parts)
    return str(response)

def count_words_approx_tokens(text):
    # Standard approximation ~ 1.3 tokens per word for Portuguese/English mix
    words = text.split()
    return int(len(words) * 1.3)

def run_benchmark(enable_speculative: bool, label: str):
    print(f"\n=======================================================")
    print(f" Inicializando Engine: {label} (Speculative={enable_speculative})")
    print(f"=======================================================")
    
    t0_init = time.perf_counter()
    engine = Engine(
        model_path=MODEL_PATH,
        backend=interfaces.GPU(),
        enable_speculative_decoding=enable_speculative,
        max_num_tokens=2048,
    )
    t1_init = time.perf_counter()
    print(f"[+] Engine inicializada em {t1_init - t0_init:.2f}s")
    
    results = []
    
    # Warmup
    print("[*] Aquecendo pipeline...")
    conv = engine.create_conversation()
    conv.send_message("Ola")
    
    for item in TEST_PROMPTS:
        p_name = item["name"]
        prompt = item["prompt"]
        print(f"\n[*] Executando prompt: '{p_name}'...")
        
        conv = engine.create_conversation()
        t_start = time.perf_counter()
        resp = conv.send_message(prompt)
        t_end = time.perf_counter()
        
        elapsed = t_end - t_start
        text = extract_text(resp)
        tokens = count_words_approx_tokens(text)
        tps = tokens / elapsed if elapsed > 0 else 0
        
        print(f"    -> Tempo: {elapsed:.3f}s | Tokens (aprox): {tokens} | TPS: {tps:.2f} tok/s")
        print(f"    -> Resumo da resposta: {text[:80].strip()}...")
        
        results.append({
            "prompt": p_name,
            "elapsed_s": elapsed,
            "tokens": tokens,
            "tps": tps,
            "sample": text[:120]
        })
        
    engine.close()
    return results

def main():
    print("[+] INICIANDO BENCHMARK COMPARATIVO NA RTX 2060")
    
    # 1. Baseline: Standard Autoregressive Decode
    res_baseline = run_benchmark(enable_speculative=False, label="Baseline (Autoregressive Padrao)")
    
    # 2. Speculative: MTP Drafter Enabled
    res_speculative = run_benchmark(enable_speculative=True, label="Transdutor Especulativo (MTP Drafter)")
    
    # 3. Summary & Comparison
    print("\n\n=======================================================")
    print(" RESULTADOS CONSOLIDADOS DO EXPERIMENTO 002")
    print("=======================================================")
    print(f"{'Prompt':<25} | {'Baseline TPS':<14} | {'MTP Spec TPS':<14} | {'Speedup':<8}")
    print("-" * 70)
    
    speedups = []
    for b, s in zip(res_baseline, res_speculative):
        sp = s["tps"] / b["tps"] if b["tps"] > 0 else 1.0
        speedups.append(sp)
        print(f"{b['prompt']:<25} | {b['tps']:>10.2f} t/s | {s['tps']:>10.2f} t/s | {sp:>6.2f}x")
        
    avg_speedup = sum(speedups) / len(speedups)
    print("-" * 70)
    print(f"Speedup Medio do Transdutor Especulativo: {avg_speedup:.2f}x")
    
    # Save JSON summary
    summary = {
        "device": "NVIDIA GeForce RTX 2060",
        "model": "gemma-4-E2B-it.litertlm",
        "baseline": res_baseline,
        "speculative": res_speculative,
        "avg_speedup": avg_speedup
    }
    with open("experiments/002_progressive_transducer/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n[+] Resultados salvos em experiments/002_progressive_transducer/benchmark_results.json")

if __name__ == "__main__":
    main()
