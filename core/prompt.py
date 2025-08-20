"""
Prompt templates that force JSON-only output from LLM providers.
"""

def make_prompt(entity: str, category_hint: str) -> str:
    """
    Create a prompt that forces JSON-only output with strict schema compliance.
    
    Args:
        entity: The entity to analyze (e.g., "Vuori")
        category_hint: Category context (e.g., "consumer apparel / activewear")
        
    Returns:
        Complete prompt string that enforces JSON output
    """
    prompt = f"""Analyze the visibility of "{entity}" in the {category_hint} space across LLM knowledge bases.

DEFINITION OF VISIBILITY:
- Breadth: How widely known across different AI models and contexts
- Freshness: How current and up-to-date the information is
- Depth/Accuracy: Level of detailed, accurate information available
- Hallucination penalty: Deduct points for inconsistent or made-up information

SCORING METHODOLOGY (0-100 for each):
- Recognition: How well LLMs recognize and identify this entity
- Media: Coverage in news, articles, and media mentions
- Context: Understanding of industry position and relationships
- Competitors: Awareness of alternatives and competitive landscape
- Consistency: Stability and agreement across different queries/models

You MUST return ONLY valid minified JSON that matches this exact schema. No markdown, no code fences, no commentary:

{{"entity":"{entity}","category":"{category_hint}","overall_score":85,"breakdown":{{"recognition":80,"media":75,"context":85,"competitors":90,"consistency":85}},"notes":"Brief analysis paragraph under 600 chars","sources":["domain1.com","brief-citation-2","source-3"]}}

CRITICAL REQUIREMENTS:
- Return ONLY the JSON object, nothing else
- All scores must be integers 0-100
- Notes must be under 600 characters
- Sources should be domains or brief citations (max 8)
- If uncertain, provide best-effort estimates and keep internal consistency
- Calculate overall_score as weighted average: recognition(30%) + media(25%) + context(20%) + consistency(15%) + competitors(10%)

Analyze "{entity}" now:"""
    
    return prompt 