# AGENTS.md: Documentation Directives (`docs/`)

This directory maintains long-form technical references, architectural specifications, and hardware routing guides.

---

## 1. Documentation Principles

1. **Rigor & Ground Truths**: All architectural claims must reflect verified code paths in Google Dawn, Direct3D 12, or LiteRT-LM.
2. **Language**: Maintain all technical documentation in **English**.
3. **Diagrams**: Use standard Mermaid diagrams (`flowchart TD`, `sequenceDiagram`) rather than ASCII art or raster images where possible.

---

## 2. Directory Index

- `architecture.md`: Complete Direct3D 12 and Dawn runtime execution model.
- `benchmarks.md`: Comparative hardware benchmark tables (RTX 2060 vs. GTX 1050 Ti vs. CPU).
- `gpu_routing.md`: Multi-GPU isolation mechanisms on Windows.
