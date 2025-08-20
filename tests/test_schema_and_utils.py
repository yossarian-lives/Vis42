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
            "sources": ["tesla.com", "techcrunch.com/tesla"]
        }
        
        assert validate_result(invalid_result) == False
    
    def test_invalid_score_range_fails(self):
        """Test that scores outside 0-100 range fail validation"""
        invalid_result = {
            "entity": "Tesla",
            "category": "automotive",
            "overall_score": 150,  # Invalid score
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
        
        assert validate_result(invalid_result) == False
    
    def test_missing_breakdown_field_fails(self):
        """Test that missing breakdown fields fail validation"""
        invalid_result = {
            "entity": "Tesla",
            "category": "automotive",
            "overall_score": 85,
            "breakdown": {
                "recognition": 90,
                "media": 80,
                # Missing context, competitors, consistency
            },
            "notes": "Tesla is well-known in the automotive space.",
            "sources": ["tesla.com", "techcrunch.com/tesla"]
        }
        
        assert validate_result(invalid_result) == False
    
    def test_notes_too_long_fails(self):
        """Test that notes exceeding 600 characters fail validation"""
        long_notes = "x" * 601  # 601 characters
        invalid_result = {
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
            "notes": long_notes,
            "sources": ["tesla.com", "techcrunch.com/tesla"]
        }
        
        assert validate_result(invalid_result) == False
    
    def test_too_many_sources_fails(self):
        """Test that more than 8 sources fail validation"""
        many_sources = [f"source{i}.com" for i in range(9)]  # 9 sources
        invalid_result = {
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
            "sources": many_sources
        }
        
        assert validate_result(invalid_result) == False

class TestFallbackResults:
    """Test fallback result generation"""
    
    def test_fallback_result_structure(self):
        """Test that fallback results have correct structure"""
        fallback = get_fallback_result("TestEntity", "Test reason")
        
        assert fallback["entity"] == "TestEntity"
        assert fallback["category"] == "unknown"
        assert fallback["overall_score"] == 40
        assert "breakdown" in fallback
        assert "notes" in fallback
        assert "sources" in fallback
        assert validate_result(fallback) == True
    
    def test_fallback_result_reason(self):
        """Test that fallback reason is included in notes"""
        reason = "API call failed"
        fallback = get_fallback_result("TestEntity", reason)
        
        assert reason in fallback["notes"]

class TestJSONUtils:
    """Test JSON coercion utilities"""
    
    def test_clean_json_parses(self):
        """Test that clean JSON parses directly"""
        clean_json = '{"key": "value", "number": 42}'
        result = coerce_json(clean_json)
        
        assert result == {"key": "value", "number": 42}
    
    def test_markdown_code_fence_extraction(self):
        """Test extraction from markdown code fences"""
        markdown_text = """
        Here's some analysis:
        
        ```json
        {"entity": "Tesla", "score": 85}
        ```
        
        Hope this helps!
        """
        result = coerce_json(markdown_text)
        
        assert result == {"entity": "Tesla", "score": 85}
    
    def test_extra_text_handling(self):
        """Test handling of text before/after JSON"""
        messy_text = "I analyzed this and found: {\"result\": \"success\"} which is great!"
        result = coerce_json(messy_text)
        
        assert result == {"result": "success"}
    
    def test_single_quote_fixing(self):
        """Test fixing of single quotes to double quotes"""
        single_quote_json = "{'key': 'value', 'number': 42}"
        result = coerce_json(single_quote_json)
        
        assert result == {"key": "value", "number": 42}
    
    def test_invalid_json_returns_none(self):
        """Test that invalid JSON returns None"""
        invalid_json = "This is not JSON at all"
        result = coerce_json(invalid_json)
        
        assert result is None

class TestEntityNormalization:
    """Test entity name normalization"""
    
    def test_vuori_variations(self):
        """Test various misspellings of Vuori"""
        assert normalize_entity("vouri") == "Vuori"
        assert normalize_entity("voui") == "Vuori"
        assert normalize_entity("vuouri") == "Vuori"
        assert normalize_entity("VOURI") == "Vuori"
    
    def test_common_entities(self):
        """Test normalization of common entity names"""
        assert normalize_entity("tesla") == "Tesla"
        assert normalize_entity("apple") == "Apple"
        assert normalize_entity("microsoft") == "Microsoft"
    
    def test_whitespace_handling(self):
        """Test handling of extra whitespace"""
        assert normalize_entity("  Tesla  ") == "Tesla"
        assert normalize_entity("Apple\n") == "Apple"
    
    def test_empty_input(self):
        """Test handling of empty/None input"""
        assert normalize_entity("") == ""
        assert normalize_entity(None) == ""
    
    def test_casing_normalization(self):
        """Test proper casing normalization"""
        assert normalize_entity("APPLE") == "Apple"
        assert normalize_entity("microsoft") == "Microsoft"

class TestOrchestrator:
    """Test orchestrator functions"""
    
    def test_calculate_overall_score(self):
        """Test weighted score calculation"""
        breakdown = {
            "recognition": 80,    # 30% = 24
            "media": 60,          # 25% = 15
            "context": 70,        # 20% = 14
            "consistency": 90,    # 15% = 13.5
            "competitors": 50     # 10% = 5
        }
        
        expected = int(round(24 + 15 + 14 + 13.5 + 5))
        result = calculate_overall_score(breakdown)
        
        assert result == expected
    
    def test_merge_results_single_provider(self):
        """Test merging when only one provider responds"""
        single_result = {
            "entity": "Tesla",
            "category": "automotive",
            "overall_score": 85,
            "breakdown": {"recognition": 90, "media": 80, "context": 85, "competitors": 75, "consistency": 85},
            "notes": "Single analysis",
            "sources": ["tesla.com"]
        }
        
        merged = merge_results([single_result])
        
        assert merged == single_result
    
    def test_merge_results_multiple_providers(self):
        """Test merging multiple provider results"""
        result1 = {
            "entity": "Tesla",
            "category": "automotive",
            "overall_score": 80,
            "breakdown": {"recognition": 85, "media": 75, "context": 80, "competitors": 70, "consistency": 80},
            "notes": "Analysis 1",
            "sources": ["tesla.com"]
        }
        
        result2 = {
            "entity": "Tesla",
            "category": "automotive",
            "overall_score": 90,
            "breakdown": {"recognition": 95, "media": 85, "context": 90, "competitors": 80, "consistency": 90},
            "notes": "Analysis 2",
            "sources": ["techcrunch.com"]
        }
        
        merged = merge_results([result1, result2])
        
        # Check that entity and category are preserved
        assert merged["entity"] == "Tesla"
        assert merged["category"] == "automotive"
        
        # Check that notes are merged
        assert "Analysis 1" in merged["notes"]
        assert "Analysis 2" in merged["notes"]
        
        # Check that sources are combined
        assert len(merged["sources"]) == 2
        assert "tesla.com" in merged["sources"]
        assert "techcrunch.com" in merged["sources"]
    
    def test_merge_results_empty_list(self):
        """Test merging empty results list"""
        fallback = merge_results([])
        
        assert fallback["entity"] == "unknown"
        assert "No provider responded" in fallback["notes"]

if __name__ == "__main__":
    # Run tests directly if file is executed
    pytest.main([__file__, "-v"]) 