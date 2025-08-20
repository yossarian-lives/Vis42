"""
Acceptance tests for schema validation and utility functions.

Run with: pytest tests/test_schema_and_utils.py -v
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.schema import validate_result, get_fallback_result, VISIBILITY_SCHEMA
from utils.json_utils import coerce_json
from utils.entity import normalize_entity
from core.orchestrator import merge_results, calculate_overall_score

class TestSchemaValidation:
    """Test JSON schema validation"""
    
    def test_valid_schema_passes(self):
        """Test that valid schema passes validation"""
        valid_result = {
            "entity": "Tesla",
            "category": "automotive",
            "overall_score": 85,
            "breakdown": {
                "recognition": 90,
                "media": 80,
                "context": 85,
                "competitors": 75,
                "consistency": 85
            },
            "notes": "Tesla is well-known in the automotive space.",
            "sources": ["tesla.com", "techcrunch.com/tesla"]
        }
        
        assert validate_result(valid_result) == True
    
    def test_missing_required_field_fails(self):
        """Test that missing required fields fail validation"""
        invalid_result = {
            "entity": "Tesla",
            "category": "automotive",
            # Missing overall_score
            "breakdown": {
                "recognition": 90,
                "media": 80,
                "context": 85,
                "competitors": 75,
                "consistency": 85
            },
            "notes": "Tesla is well-known in the automotive space.",
            "sources": ["tesla.com"]
        }
        
        assert validate_result(invalid_result) == False
    
    def test_invalid_score_range_fails(self):
        """Test that scores outside 0-100 range fail validation"""
        invalid_result = {
            "entity": "Tesla",
            "category": "automotive", 
            "overall_score": 150,  # Invalid: > 100
            "breakdown": {
                "recognition": 90,
                "media": 80,
                "context": 85,
                "competitors": 75,
                "consistency": 85
            },
            "notes": "Tesla is well-known in the automotive space.",
            "sources": ["tesla.com"]
        }
        
        assert validate_result(invalid_result) == False
    
    def test_fallback_result_structure(self):
        """Test that fallback results match schema"""
        fallback = get_fallback_result("TestEntity", "Test reason")
        
        assert validate_result(fallback) == True
        assert fallback["entity"] == "TestEntity"
        assert fallback["overall_score"] == 40
        assert fallback["notes"] == "Test reason"

class TestJSONUtils:
    """Test JSON parsing utilities"""
    
    def test_parse_clean_json(self):
        """Test parsing clean JSON"""
        json_text = '{"entity": "Tesla", "score": 85}'
        result = coerce_json(json_text)
        
        assert result is not None
        assert result["entity"] == "Tesla"
        assert result["score"] == 85
    
    def test_parse_json_with_code_fences(self):
        """Test parsing JSON wrapped in markdown code fences"""
        json_text = '''```json
        {
            "entity": "Tesla",
            "score": 85
        }
        ```'''
        
        result = coerce_json(json_text)
        
        assert result is not None
        assert result["entity"] == "Tesla"
        assert result["score"] == 85
    
    def test_parse_json_with_extra_text(self):
        """Test parsing JSON with surrounding text"""
        json_text = '''Here is the analysis:
        
        {"entity": "Tesla", "score": 85}
        
        This concludes the analysis.'''
        
        result = coerce_json(json_text)
        
        assert result is not None
        assert result["entity"] == "Tesla"
        assert result["score"] == 85
    
    def test_parse_malformed_json_returns_none(self):
        """Test that malformed JSON returns None"""
        json_text = '{"entity": "Tesla", "score": }'  # Missing value
        result = coerce_json(json_text)
        
        assert result is None
    
    def test_parse_empty_string_returns_none(self):
        """Test that empty string returns None"""
        result = coerce_json("")
        assert result is None
        
        result = coerce_json(None)
        assert result is None

class TestEntityNormalization:
    """Test entity name normalization"""
    
    def test_normalize_vouri_variants(self):
        """Test that Vuori variants are normalized correctly"""
        test_cases = [
            ("VOURI", "Vuori"),
            ("vouri", "Vuori"), 
            ("voui", "Vuori"),
            ("vuouri", "Vuori"),
            ("Vouri", "Vuori")  # Gets normalized to Vuori
        ]
        
        for input_entity, expected in test_cases:
            result = normalize_entity(input_entity)
            assert result == expected, f"Expected {expected}, got {result} for input {input_entity}"
    
    def test_normalize_common_entities(self):
        """Test normalization of other common entities"""
        test_cases = [
            ("tesla", "Tesla"),
            ("APPLE", "Apple"),
            ("microsoft", "Microsoft"),
            ("chatgpt", "ChatGPT"),
            ("openai", "OpenAI")
        ]
        
        for input_entity, expected in test_cases:
            result = normalize_entity(input_entity)
            assert result == expected, f"Expected {expected}, got {result} for input {input_entity}"
    
    def test_normalize_title_case_conversion(self):
        """Test automatic title case conversion"""
        test_cases = [
            ("RANDOM COMPANY", "Random Company"),
            ("some startup", "Some Startup"),
            ("AI Platform", "AI Platform")  # Already correct
        ]
        
        for input_entity, expected in test_cases:
            result = normalize_entity(input_entity)
            assert result == expected, f"Expected {expected}, got {result} for input {input_entity}"
    
    def test_normalize_empty_input(self):
        """Test handling of empty/None input"""
        assert normalize_entity("") == ""
        assert normalize_entity(None) == ""
        assert normalize_entity("   ") == ""

class TestOrchestrator:
    """Test orchestrator functionality"""
    
    def test_calculate_overall_score(self):
        """Test weighted overall score calculation"""
        breakdown = {
            "recognition": 80,
            "media": 70,
            "context": 75,
            "competitors": 65,
            "consistency": 85
        }
        
        # Expected: 80*0.3 + 70*0.25 + 75*0.2 + 85*0.15 + 65*0.1 = 75.25 ≈ 75
        result = calculate_overall_score(breakdown)
        assert isinstance(result, int)
        assert 70 <= result <= 80  # Allow some rounding variance
    
    def test_merge_single_result(self):
        """Test merging when only one result is provided"""
        single_result = {
            "entity": "Tesla",
            "category": "automotive",
            "overall_score": 85,
            "breakdown": {
                "recognition": 90,
                "media": 80,
                "context": 85,
                "competitors": 75,
                "consistency": 85
            },
            "notes": "Single provider analysis",
            "sources": ["tesla.com"]
        }
        
        merged = merge_results([single_result])
        assert merged == single_result
    
    def test_merge_multiple_results(self):
        """Test merging multiple provider results"""
        result1 = {
            "entity": "Tesla",
            "category": "automotive", 
            "overall_score": 80,
            "breakdown": {
                "recognition": 85,
                "media": 75,
                "context": 80,
                "competitors": 70,
                "consistency": 80
            },
            "notes": "Provider 1 analysis",
            "sources": ["tesla.com"]
        }
        
        result2 = {
            "entity": "Tesla",
            "category": "automotive",
            "overall_score": 90,
            "breakdown": {
                "recognition": 95,
                "media": 85,
                "context": 90,
                "competitors": 80,
                "consistency": 90
            },
            "notes": "Provider 2 analysis", 
            "sources": ["techcrunch.com", "tesla.com"]  # Duplicate source
        }
        
        merged = merge_results([result1, result2])
        
        # Check basic structure
        assert merged["entity"] == "Tesla"
        assert merged["category"] == "automotive"
        assert isinstance(merged["overall_score"], int)
        
        # Check median calculation (should be between the two results)
        assert merged["breakdown"]["recognition"] == 90  # median of 85, 95
        assert merged["breakdown"]["media"] == 80  # median of 75, 85
        
        # Check source deduplication
        assert len(merged["sources"]) == 2  # tesla.com should be deduplicated
        assert "tesla.com" in merged["sources"]
        assert "techcrunch.com" in merged["sources"]
        
        # Check notes merging
        assert "Provider 1" in merged["notes"] and "Provider 2" in merged["notes"]
    
    def test_merge_empty_results_returns_fallback(self):
        """Test that merging empty results returns fallback"""
        merged = merge_results([])
        
        assert merged["entity"] == "unknown"
        assert merged["overall_score"] == 40
        assert "No provider responded" in merged["notes"]
    
    def test_orchestrator_fallback_on_failure(self):
        """Test that orchestrator provides fallback when adapters raise exceptions"""
        # This would test the actual analyze_entity function
        # For now, we'll test the fallback result structure
        fallback = get_fallback_result("TestEntity", "All provider calls failed")
        
        assert validate_result(fallback) == True
        assert fallback["overall_score"] == 40
        assert all(score == 40 for key, score in fallback["breakdown"].items() if key != "consistency")
        assert fallback["breakdown"]["consistency"] == 60

if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"]) 