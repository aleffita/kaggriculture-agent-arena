"""LiteRT-LM Market Arbitrageur: H24 Daily Macro Policy on NVIDIA GTX 1050 Ti."""

from kaggriculture.agents.personality_agent import make_personality_agent

agent = make_personality_agent("market_arbitrageur", horizon=24, is_hybrid=False)
