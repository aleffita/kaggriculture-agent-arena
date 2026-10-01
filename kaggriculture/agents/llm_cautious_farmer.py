"""LiteRT-LM Cautious Farmer: H24 Daily Macro Policy on NVIDIA GTX 1050 Ti."""

from kaggriculture.agents.personality_agent import make_personality_agent

agent = make_personality_agent("cautious_farmer", horizon=24, is_hybrid=False)
