# AGENTS.md: Research Subsystem & Epistemic Ledger

Welcome to the `research/` directory. This subsystem acts as the centralized epistemological hub and empirical memory for the entire repository.

---

## 1. Subsystem Purpose

While `experiments/` hosts executable probes, benchmark scripts, and code artifacts, `research/` consolidates the **knowledge, hardware characterizations, and operational lessons** gained across all investigations.

Whenever an experiment yields new metrics, architectural surprises, or hardware anomalies, they MUST be logged into `research/RESEARCH_LEDGER.md`.

---

## 2. Directory Hierarchy

```text
research/
├── AGENTS.md               # This directive document
├── RESEARCH_LEDGER.md      # The master chronological & tagged ledger of insights
├── hardware_pascal_sm61.md # Architectural characterization of GP107 (GTX 1050 Ti)
└── dxgi_direct3d12_shim.md # Direct3D 12 adapter hooking & multi-GPU routing theory
```

---

## 3. Rules for Contributing to `RESEARCH_LEDGER.md`

Every new ledger entry must follow the standardized ledger entry template:

```markdown
### [LEDGER-XXX] Title of Insight or Finding
- **Date**: YYYY-MM-DD
- **Target Subsystem / Hardware**: e.g. `Pascal GP107 / D3D12 / LiteRT-LM`
- **Related Experiment**: `experiments/00X_...`
- **Empirical Observation**: Concrete measurement (t/s, MB, ms, acceptance %).
- **Underlying Mechanism**: Why this occurs at the driver, GPU microarchitecture, or runtime boundary.
- **Operational Directive**: The prescriptive rule that future agents and code must adhere to.
```

---

## 4. Ground Truths to Respect

1. All documentation in `research/` must be written in **English**.
2. Avoid vague qualitative claims ("runs fast", "feels smooth"); provide **exact measured numbers** with hardware specifications.
3. Keep the ledger append-only and dialectically refined—never delete previous observations, update them with new evidence.
