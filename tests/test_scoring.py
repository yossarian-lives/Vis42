from src.llm_visibility.scoring import score

def test_score_monotonic():
    """Test that scores are monotonic and within expected ranges"""
    base = {
        "recognition_score": 0.6,
        "detail_score": 0.5,
        "context_score": 0.4,
        "consistency_score": 0.7,
        "competitors": ["A"]
    }
    s = score(base)
    assert 0 < s["Overall"] <= 100
    assert s["Recognition"] == 60.0
    assert s["Detail"] == 50.0
    assert s["Context"] == 40.0
    assert s["Competitors"] == 100.0
    assert s["Consistency"] == 70.0

def test_score_no_competitors():
    """Test scoring when no competitors are provided"""
    base = {
        "recognition_score": 0.8,
        "detail_score": 0.7,
        "context_score": 0.6,
        "consistency_score": 0.9
    }
    s = score(base)
    assert s["Competitors"] == 0.0
    assert s["Overall"] > 0

def test_score_edge_cases():
    """Test edge cases for scoring"""
    # All zeros
    zero_scores = {
        "recognition_score": 0.0,
        "detail_score": 0.0,
        "context_score": 0.0,
        "consistency_score": 0.0
    }
    s = score(zero_scores)
    assert s["Overall"] == 0.0
    
    # All ones (perfect scores)
    perfect_scores = {
        "recognition_score": 1.0,
        "detail_score": 1.0,
        "context_score": 1.0,
        "consistency_score": 1.0,
        "competitors": ["A", "B"]
    }
    s = score(perfect_scores)
    assert s["Overall"] > 90  # Should be very high with consistency multiplier

def test_score_missing_fields():
    """Test scoring when some fields are missing"""
    partial = {
        "recognition_score": 0.5,
        "detail_score": 0.6
        # Missing context_score, consistency_score, competitors
    }
    s = score(partial)
    assert s["Context"] == 0.0
    assert s["Consistency"] == 0.0
    assert s["Competitors"] == 0.0
    assert s["Overall"] > 0  # Should still calculate with available scores 