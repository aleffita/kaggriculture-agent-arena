# Inicia o servidor do Dashboard de Replays e Sandbox de Submissão Kaggle
$env:PYTHONPATH = ".;kaggriculture;kaggriculture/agents"
uv run python -m kaggriculture.dashboard.app
