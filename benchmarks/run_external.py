"""
Unified Third-Party External Benchmark Runner for LiteRT-LM & Unified CED Runtime.
Wraps EleutherAI lm-evaluation-harness entrypoints with automatic server verification,
Windows UTF-8 encoding normalization, and structured JSON report persistence.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output encoding across Windows consoles to prevent charmap errors (e.g. \u2191)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
except ImportError:
    console = None


SUITE_TASK_MAP: Dict[str, List[str]] = {
    "math": ["gsm8k", "minerva_math_algebra"],
    "code": ["humaneval", "mbpp"],
    "multimodal": ["chartqa"],
    "all": ["gsm8k", "minerva_math_algebra", "humaneval", "mbpp", "chartqa"],
}


def print_msg(msg: str, style: str = "bold green") -> None:
    if console:
        console.print(f"[{style}]{msg}[/{style}]")
    else:
        print(msg)


def print_err(msg: str) -> None:
    if console:
        console.print(f"[bold red]ERROR: {msg}[/bold red]")
    else:
        print(f"ERROR: {msg}", file=sys.stderr)


def check_server_health(base_url: str, timeout: float = 3.0) -> bool:
    """Verifica se o servidor OpenAI-compatível local está respondendo."""
    prefix = base_url.split("/v1")[0]
    for test_path in ["/health", "/v1/models", "/"]:
        try:
            req = urllib.request.Request(prefix + test_path, headers={"User-Agent": "litert-bench-external"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status in (200, 204):
                    return True
        except urllib.error.HTTPError as e:
            if e.code in (200, 204, 405):
                return True
        except Exception:
            pass
    return False


def run_benchmark_task(
    task_name: str,
    model_name: str,
    base_url: str,
    api_key: str,
    limit: Optional[int],
    output_dir: Path,
    confirm_unsafe_code: bool = True
) -> Dict[str, Any]:
    """Invoca o lm-eval programaticamente para uma tarefa específica."""
    from lm_eval.evaluator import simple_evaluate

    print_msg(f"\n[bold cyan]▶ Iniciando avaliação externa: {task_name} (limite: {limit} amostras)...[/bold cyan]")
    t0 = time.perf_counter()

    model_args = {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
        "num_concurrent": 1,
    }

    try:
        eval_results = simple_evaluate(
            model="local-chat-completions",
            model_args=model_args,
            tasks=[task_name],
            limit=limit,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            confirm_run_unsafe_code=confirm_unsafe_code,
            log_samples=True,
        )
    except Exception as e:
        print_err(f"Falha ao executar tarefa '{task_name}': {e}")
        return {"task": task_name, "status": "error", "error": str(e)}

    elapsed_s = time.perf_counter() - t0

    # Persistir relatório JSON em reports/
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = output_dir / f"lm_eval_{task_name}_{timestamp_str}.json"
    
    # Converter para dicionário serializável
    if hasattr(eval_results, "to_dict"):
        results_dict = eval_results.to_dict()
    elif isinstance(eval_results, dict):
        results_dict = eval_results
    else:
        results_dict = {
            "results": getattr(eval_results, "results", {}),
            "configs": getattr(eval_results, "configs", {}),
            "versions": getattr(eval_results, "versions", {}),
            "n-shot": getattr(eval_results, "n-shot", {}),
        }

    results_dict["elapsed_time_seconds"] = round(elapsed_s, 2)
    results_dict["timestamp"] = timestamp_str
    results_dict["model_name"] = model_name
    results_dict["base_url"] = base_url

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)

    print_msg(f"✔ Relatório salvo com sucesso em: {report_file} ({elapsed_s:.1f}s)", "bold green")
    return results_dict


def render_summary_table(all_results: List[Dict[str, Any]]) -> None:
    """Renderiza tabela executiva de métricas coletadas."""
    if not console:
        for res in all_results:
            print(f"Tarefa: {res.get('task') or list(res.get('results', {}).keys())} - Duração: {res.get('elapsed_time_seconds')}s")
        return

    table = Table(title="[bold yellow]Resultados dos Benchmarks Externos (EleutherAI lm-eval)[/bold yellow]", show_header=True)
    table.add_column("Tarefa / Benchmark", style="cyan", no_wrap=True)
    table.add_column("Amostras", style="magenta")
    table.add_column("Métrica Principal", style="green")
    table.add_column("Score / Acurácia", style="bold white")
    table.add_column("Tempo (s)", style="blue")
    table.add_column("Status", style="bold green")

    for entry in all_results:
        task_data = entry.get("results", {})
        if not task_data:
            table.add_row(
                str(entry.get("task", "desconhecido")),
                "-",
                "-",
                "-",
                str(entry.get("elapsed_time_seconds", "-")),
                "[red]ERRO[/red]"
            )
            continue

        for task_name, metrics in task_data.items():
            sample_len = metrics.get("sample_len", metrics.get("samples", "-"))
            # Buscar métrica representativa
            score_str = "-"
            metric_name = "-"
            for k, v in metrics.items():
                if any(m in k for m in ["exact_match", "acc", "relaxed_accuracy", "pass@1", "math_verify"]):
                    if "stderr" not in k:
                        metric_name = k
                        score_str = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                        break

            table.add_row(
                task_name,
                str(sample_len),
                metric_name,
                score_str,
                f"{entry.get('elapsed_time_seconds', 0.0):.2f}",
                "[green]SUCESSO[/green]"
            )

    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Runner unificado para benchmarks externos de terceiros (lm-eval, math, code, multimodal)."
    )
    parser.add_argument(
        "--suite",
        choices=["math", "code", "multimodal", "all"],
        default="all",
        help="Conjunto de tarefas a executar (default: all)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Lista de tarefas específicas separadas por vírgula (ex: gsm8k,minerva_math_algebra,mbpp)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite de amostras por tarefa (default: None para suíte completa sem limite)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-20b",
        help="Identificador do modelo no endpoint (default: gpt-oss-20b)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8765/v1/chat/completions",
        help="URL base do endpoint OpenAI chat completions"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="sk-unified-ced-local",
        help="Chave de API local"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/reports",
        help="Diretório onde salvar os relatórios JSON"
    )
    parser.add_argument(
        "--no-server-check",
        action="store_true",
        help="Pular verificação prévia de conectividade do servidor"
    )

    args = parser.parse_args()
    raw_out = Path(args.output_dir)
    output_dir = raw_out if raw_out.is_absolute() else (Path(__file__).parent / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verificação de Saúde do Servidor
    if not args.no_server_check:
        print_msg(f"Verificando conectividade com o servidor em {args.base_url}...", "bold yellow")
        if not check_server_health(args.base_url):
            print_err(
                f"Servidor inacessível em {args.base_url}!\n"
                "Certifique-se de que o runtime headless está ativo:\n"
                "  uv run litert-web --host 127.0.0.1 --port 8765 --model gpt-oss --headless"
            )
            return 1
        print_msg("✔ Servidor online e responsivo!", "bold green")

    # 2. Definição da lista de tarefas
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = SUITE_TASK_MAP.get(args.suite, SUITE_TASK_MAP["all"])

    effective_limit = None if (args.limit is None or args.limit <= 0) else args.limit
    limit_str = f"{effective_limit} amostras" if effective_limit else "COMPLETO (Sem Limite)"

    print_msg(f"\n[bold magenta]══════════════════════════════════════════════════════════[/bold magenta]")
    print_msg(f" SUÍTE DE AVALIAÇÃO EXTERNA: {args.suite.upper()} ({len(tasks)} tarefas)")
    print_msg(f" Modelo: {args.model} | Modo: {limit_str} | Endpoint: {args.base_url}")
    print_msg(f"[bold magenta]══════════════════════════════════════════════════════════[/bold magenta]")

    all_results: List[Dict[str, Any]] = []

    for task_name in tasks:
        res = run_benchmark_task(
            task_name=task_name,
            model_name=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            limit=effective_limit,
            output_dir=output_dir,
            confirm_unsafe_code=True
        )
        all_results.append(res)

    print_msg("\n[bold green]✔ Todos os benchmarks externos foram finalizados![/bold green]")
    render_summary_table(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
