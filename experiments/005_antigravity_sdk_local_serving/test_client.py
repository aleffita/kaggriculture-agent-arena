"""Concurrent test client for local LiteRT-LM OpenAI server."""

import concurrent.futures
import json
import time
import requests
from rich.console import Console

console = Console()

SERVER_URL = "http://127.0.0.1:9379/v1/chat/completions"

def send_chat_completion(user_id: int, prompt: str):
    payload = {
        "model": "gemma-4-E2B-it.litertlm",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "max_tokens": 128,
    }
    start = time.perf_counter()
    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=60)
        elapsed = time.perf_counter() - start
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {
                "user_id": user_id,
                "status": "OK",
                "elapsed": elapsed,
                "response": content[:80].replace("\n", " "),
            }
        else:
            return {"user_id": user_id, "status": f"HTTP {resp.status_code}", "elapsed": elapsed, "response": resp.text}
    except Exception as e:
        return {"user_id": user_id, "status": "ERROR", "elapsed": time.perf_counter() - start, "response": str(e)}

def run_concurrent_test(num_users: int = 3):
    console.print(f"[bold cyan]=== Testing Multi-User Concurrency ({num_users} simultaneous users) ===[/bold cyan]\n")
    prompts = [
        "What is the capital of France?",
        "Explain Ohm's law in one sentence.",
        "What are the primary colors in additive mixing?",
    ]

    start_total = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = [
            executor.submit(send_chat_completion, i, prompts[i % len(prompts)])
            for i in range(num_users)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    total_elapsed = time.perf_counter() - start_total

    for r in results:
        status_color = "green" if r["status"] == "OK" else "red"
        console.print(f"  User {r['user_id']}: [{status_color}]{r['status']}[/{status_color}] in {r['elapsed']:.2f}s | Reply: {r['response']}...")

    console.print(f"\n[bold green]Total concurrent batch completed in {total_elapsed:.2f}s[/bold green]")

if __name__ == "__main__":
    run_concurrent_test(3)
