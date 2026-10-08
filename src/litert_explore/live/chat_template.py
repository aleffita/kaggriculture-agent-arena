"""
Jinja2 Chat Template Renderer for Unified CED Runtime.
Supports OpenAI Harmony (<|start|>{header}<|message|>{content}<|end|>),
ChatML, Llama-3, and custom user-provided Jinja templates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import jinja2


# Template canônico do OpenAI Harmony (utilizado por GPT-OSS / GPT-o1)
HARMONY_JINJA_TEMPLATE = """{% for message in messages %}<|start|>{{ message.role }}{% if message.channel %}<|channel|>{{ message.channel }}{% endif %}{% if message.recipient %}<|recipient|>{{ message.recipient }}{% endif %}<|message|>{{ message.content }}<|end|>
{% endfor %}{% if add_generation_prompt %}<|start|>assistant<|channel|>analysis<|message|>{% endif %}"""

# Template canônico ChatML (padrão indústria)
CHATML_JINJA_TEMPLATE = """{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

# Template canônico Llama-3 / Instruct
LLAMA3_JINJA_TEMPLATE = """{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>

{{ message.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}"""


class ChatTemplateRenderer:
    """Renderizador flexível de chat templates baseado em Jinja2."""

    def __init__(self, default_format: str = "harmony"):
        self.default_format = default_format
        self._env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
        self._templates: Dict[str, jinja2.Template] = {
            "harmony": self._env.from_string(HARMONY_JINJA_TEMPLATE),
            "chatml": self._env.from_string(CHATML_JINJA_TEMPLATE),
            "llama3": self._env.from_string(LLAMA3_JINJA_TEMPLATE),
        }

    def render(
        self,
        messages: List[Dict[str, Any]],
        format_name: Optional[str] = None,
        add_generation_prompt: bool = True,
        custom_template: Optional[str] = None,
        **extra_kwargs: Any,
    ) -> str:
        """
        Renderiza a lista de mensagens usando Jinja2.
        Normaliza roles (developer -> system se necessário).
        """
        # Normalização de mensagens
        normalized_msgs = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content") or ""
            if isinstance(content, list):
                # Extrair blocos de texto se multimodal
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content_str = " ".join(text_parts)
            else:
                content_str = str(content)

            normalized_msgs.append({
                "role": role,
                "content": content_str,
                "channel": m.get("channel"),
                "recipient": m.get("recipient"),
            })

        if custom_template:
            tpl = self._env.from_string(custom_template)
            return tpl.render(messages=normalized_msgs, add_generation_prompt=add_generation_prompt, **extra_kwargs)

        fmt = format_name or self.default_format
        tpl = self._templates.get(fmt, self._templates["harmony"])
        return tpl.render(messages=normalized_msgs, add_generation_prompt=add_generation_prompt, **extra_kwargs)


# Instância global compartilhada
_default_renderer = ChatTemplateRenderer()


def render_chat_prompt(
    messages: List[Dict[str, Any]],
    format_name: str = "harmony",
    add_generation_prompt: bool = True,
    custom_template: Optional[str] = None,
) -> str:
    """Função utilitária para renderizar templates de chat."""
    return _default_renderer.render(
        messages=messages,
        format_name=format_name,
        add_generation_prompt=add_generation_prompt,
        custom_template=custom_template,
    )
