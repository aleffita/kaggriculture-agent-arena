"""LiteRT-LM Land Baron: H48-Hybrid Policy on NVIDIA GTX 1050 Ti."""

from kaggriculture.agents.personality_agent import make_personality_agent

agent = make_personality_agent("land_baron", horizon=48, is_hybrid=True)
