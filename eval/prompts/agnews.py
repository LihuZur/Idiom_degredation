"""Prompts for AG News evaluation (topic classification)."""

SYSTEM = (
    "You are a news topic classifier. Read the news snippet and respond with exactly one "
    "of these four categories and nothing else: World, Sports, Business, Sci/Tech."
)

USER_TEMPLATE = """News snippet: {x}
Category:"""
