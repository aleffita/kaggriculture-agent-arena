"""LiteRT-LM Melon Monopolist: H48-Hybrid Policy on NVIDIA GTX 1050 Ti."""

from kaggriculture.agents.personality_agent import make_personality_agent

agent = make_personality_agent("melon_monopolist", horizon=48, is_hybrid=True)
