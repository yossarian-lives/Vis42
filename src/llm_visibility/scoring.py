def score(response) -> dict:
    """
    response: normalized structure with fields:
      {facts: [...], competitors:[...], detail_score:0..1, context_score:0..1, recognition_score:0..1, consistency_score:0..1}
    """
    recognition = response.get("recognition_score", 0)      # 45%
    detail      = response.get("detail_score", 0)           # 25%
    context     = response.get("context_score", 0)          # 20%
    competitors = 1.0 if response.get("competitors") else 0 # 10%
    base = 0.45*recognition + 0.25*detail + 0.20*context + 0.10*competitors
    overall = 100 * base * (0.85 + 0.30*response.get("consistency_score", 0))  # multiplier
    return {
        "Recognition": 100*recognition,
        "Detail": 100*detail,
        "Context": 100*context,
        "Competitors": 100*competitors,
        "Consistency": 100*response.get("consistency_score", 0),
        "Overall": round(overall, 1),
    } 