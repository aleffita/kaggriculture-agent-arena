"""
Chat Templates, Harmony Protocol and Server-Side ABI Isolation.
Gerencia Jinja2 templates, integração com openai_harmony e isolamento estrito
entre tools internas de runtime e chamadas externas de clientes.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import jinja2

try:
    import openai_harmony
    from openai_harmony import (
        Author,
        Conversation,
        Message,
        Role,
        TextContent,
        load_harmony_encoding,
    )
    HARMONY_AVAILABLE = True
except ImportError:
    openai_harmony = None
    HARMONY_AVAILABLE = False


# Template Jinja2 canônico para Gemma
GEMMA_JINJA_TEMPLATE = """{%- for message in messages -%}
{{- '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<start_of_turn>model\n' -}}
{%- endif -%}"""

# Template Jinja2 ChatML
CHATML_JINJA_TEMPLATE = """{%- for message in messages -%}
{{- '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n' -}}
{%- endif -%}"""


def is_not_builtin_tool(recipient: str, treat_functions_python_as_builtin: bool = True) -> bool:
    """
    Verifica se o destinatário de uma chamada de tool é uma ferramenta externa do cliente
    ou uma ferramenta interna de servidor (built-in).
    Alinhado ao padrão da OpenAI em gpt-oss / responses_api:
    Ferramentas internas de servidor JAMAIS vazam como tool_calls para o cliente.
    """
    if not recipient:
        return False
    rec_lower = recipient.lower()

    if treat_functions_python_as_builtin and rec_lower in ("python", "functions.python", "python_repl"):
        return False

    # Ferramentas internas do runtime
    if (
        rec_lower.startswith("browser.")
        or rec_lower in ("python", "assistant", "system", "request_budget")
        or rec_lower.startswith("virtual_expert.")
        or rec_lower.startswith("internal_")
        or rec_lower in ("microtex_lean4", "analytical_plotter", "disk_expert_store", "llvm_jit")
    ):
        return False

    return True


def is_raw_code_completion(messages: List[Dict[str, Any]]) -> bool:
    """
    Detecta se a requisição é de completion bruto de código (ex: HumanEval, MBPP).
    Para essas requisições, o prompt NUNCA deve ser embrulhado em templates de chat,
    evitando markdown code fences (```python) que quebram com a stop sequence 'until: [\"\\ndef\"]'.
    """
    if not messages:
        return False

    # Se há apenas uma mensagem de usuário ou última mensagem do usuário
    user_msgs = [m for m in messages if m.get("role") in ("user", "human")]
    if not user_msgs:
        return False

    last_user_content = str(user_msgs[-1].get("content") or "").strip()

    # Marcadores universais de injeção direta de código
    code_starters = ("def ", "import ", "from ", "class ", "async def ")
    if last_user_content.startswith(code_starters) or "[BEGIN]" in last_user_content:
        return True

    return False


class ChatTemplateManager:
    """Gerenciador unificado de templates Jinja2 e protocolo Harmony."""

    def __init__(self):
        self._env = jinja2.Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.StrictUndefined,
        )
        self._gemma_template = self._env.from_string(GEMMA_JINJA_TEMPLATE)
        self._chatml_template = self._env.from_string(CHATML_JINJA_TEMPLATE)

    def render(
        self,
        messages: List[Dict[str, Any]],
        model_name: str = "gemma-4-E2B-it",
        add_generation_prompt: bool = True,
        system_prompt: Optional[str] = None
    ) -> str:
        """Renderiza a sequência de mensagens usando o template apropriado."""
        if is_raw_code_completion(messages):
            user_msgs = [m for m in messages if m.get("role") in ("user", "human")]
            return str(user_msgs[-1].get("content") or "")

        # Normaliza mensagens
        norm_messages: List[Dict[str, str]] = []
        has_system = False

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(parts).strip()
            else:
                content = str(content)

            if role == "system":
                has_system = True
                if system_prompt:
                    content = f"{system_prompt}\n\n{content}"
                norm_messages.append({"role": "system", "content": content})
            elif role in ("user", "human"):
                norm_messages.append({"role": "user", "content": content})
            elif role in ("assistant", "model"):
                norm_messages.append({"role": "model", "content": content})

        if system_prompt and not has_system:
            # Insere system prompt antes da primeira mensagem de usuário
            if norm_messages and norm_messages[0]["role"] == "user":
                norm_messages[0]["content"] = f"{system_prompt}\n\n{norm_messages[0]['content']}"
            else:
                norm_messages.insert(0, {"role": "system", "content": system_prompt})

        m_low = (model_name or "").lower()
        if "chatml" in m_low or "qwen" in m_low or "gpt" in m_low:
            return self._chatml_template.render(
                messages=norm_messages,
                add_generation_prompt=add_generation_prompt,
            )

        # Default para Gemma
        # Mapeia roles para Gemma (<start_of_turn>user / <start_of_turn>model)
        gemma_messages = []
        for m in norm_messages:
            r = m["role"]
            if r in ("user", "system"):
                gemma_messages.append({"role": "user", "content": m["content"]})
            else:
                gemma_messages.append({"role": "model", "content": m["content"]})

        return self._gemma_template.render(
            messages=gemma_messages,
            add_generation_prompt=add_generation_prompt,
        )


def sanitize_code_completion_continuation(prompt: str, generated: str) -> str:
    """
    Sanitiza continuações de código para benchmarks de infill (HumanEval, MBPP).
    Modelos instruídos (-it) tendem a responder com markdown fences (```python)
    e a repetir o cabeçalho/docstring da função do prompt.
    Essa função remove o invólucro de markdown e qualquer eco do prompt,
    garantindo que o retorno seja a continuação real do corpo da função.
    """
    if not generated:
        return generated

    text = generated.strip()

    # Remove markdown code fence de abertura
    if text.startswith("```python"):
        text = text[len("```python"):].lstrip()
    elif text.startswith("```"):
        text = text[3:].lstrip()

    # Remove markdown fence de fechamento
    if text.endswith("```"):
        text = text[:-3].rstrip()

    # Se o modelo ecoou o prompt inteiro
    p_stripped = prompt.strip()
    if text.startswith(p_stripped):
        text = text[len(p_stripped):]
    else:
        # Se ecoou a linha 'def ...:'
        def_lines = [l for l in p_stripped.splitlines() if l.strip().startswith("def ")]
        if def_lines:
            last_def = def_lines[-1].strip()
            if last_def in text:
                text = text.split(last_def, 1)[1]
                # Se há docstring repetida após o def
                if '"""' in text:
                    parts = text.split('"""')
                    if len(parts) >= 3:
                        text = '"""'.join(parts[2:])

    return text
