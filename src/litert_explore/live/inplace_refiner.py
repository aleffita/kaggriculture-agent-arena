"""
Chris Hay (Lazarus / LARQL) Virtual Expert In-Place Stream Refiner.
Implementa a tese de Chris Hay (chuk-lazarus / LARQL) de Virtual Experts injetados
diretamente no fluxo interno do modelo: quando uma avaliação ou equação aritmética
(ex: A + B = C) é processada, o Virtual Expert avalia a expressão em Python/SymPy.
Se C != J (onde J é o resultado verificado), o Virtual Expert aplica refinamento cirúrgico
in-place substituindo C por J, preservando rigorosamente a formatação, espaços,
markdown e pontuação original, sem interferir no prompt ou criar respostas sintéticas.
"""
from __future__ import annotations

import math
import re
from typing import Optional, Tuple

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    sympy = None
    SYMPY_AVAILABLE = False


def _safe_eval_math(expr_str: str) -> Optional[float]:
    """Avalia seguramente uma expressão aritmética simples."""
    cleaned = expr_str.strip().replace(",", "").replace("$", "")
    # Permitir apenas números, operadores básicos e parênteses
    if not re.match(r"^[\d\s\+\-\*\/\(\)\.\^]+$", cleaned):
        return None
    try:
        # Substitui ^ por ** para potência
        py_expr = cleaned.replace("^", "**")
        safe_scope = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        val = eval(py_expr, {"__builtins__": {}}, safe_scope)
        if isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val):
            return float(val)
    except Exception:
        pass
    return None


def refine_arithmetic_stream_in_place(text: str, enabled: bool = True) -> str:
    """
    Inspeciona o texto neural gerado e aplica correções cirúrgicas in-place em expressões aritméticas.
    Preserva estritamente indentação, delimitadores LaTeX ($$), e sintaxe de benchmark.
    """
    if not enabled or not text:
        return text

    last_verified_result = None

    # 1. Padrão para equações aritméticas: <expr> = <resultado_declarado>
    # Ex: "15 + 27 = 43" ou "12 * 8 = 95" ou "$15 + 27 = 43$"
    eq_pattern = re.compile(
        r"((?:(?:\$|\b))(\d+(?:\.\d+)?(?:\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)+)\s*=\s*)(\d+(?:\.\d+)?((?:\$|\b)))"
    )

    def _replace_equation(match: re.Match) -> str:
        nonlocal last_verified_result
        prefix = match.group(1)       # ex: "15 + 27 = "
        expr = match.group(2)         # ex: "15 + 27"
        claimed_str = match.group(3)  # ex: "43"
        suffix = match.group(4) or "" # ex: "$" ou ""

        claimed_num_str = claimed_str.replace(suffix, "").strip() if suffix else claimed_str.strip()
        computed_val = _safe_eval_math(expr)
        if computed_val is None:
            return match.group(0)

        # Formata resultado numérico
        if computed_val.is_integer():
            correct_str = str(int(computed_val))
        else:
            correct_str = f"{computed_val:.4f}".rstrip("0").rstrip(".")

        last_verified_result = correct_str

        # Se já estiver correto, mantém intacto
        try:
            if float(claimed_num_str) == float(correct_str):
                return match.group(0)
        except ValueError:
            pass

        # Refinamento cirúrgico in-place (Lazarus style)
        return f"{prefix}{correct_str}{suffix}"

    refined_text = eq_pattern.sub(_replace_equation, text)

    # 2. Consistência com marcador de resposta final GSM8K (#### <num>)
    if last_verified_result is not None:
        gsm_marker_pattern = re.compile(r"(####\s*)(\d+(?:\.\d+)?)")
        m = gsm_marker_pattern.search(refined_text)
        if m:
            claimed_ans = m.group(2)
            try:
                if float(claimed_ans) != float(last_verified_result):
                    # Corrige o marcador final para ser matematicamente consistente com a derivação
                    refined_text = gsm_marker_pattern.sub(rf"\g<1>{last_verified_result}", refined_text)
            except ValueError:
                pass

    return refined_text
