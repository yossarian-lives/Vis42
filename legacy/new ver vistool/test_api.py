"""
Tests for LLM Visibility API
"""
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from main import app
from schemas import AnalyzeRequest, Subscores, ProviderResponse
from scoring import compute_subscores_from_provider, visibility_from_subscores, aggregate_results
from providers import try_parse_json


client = TestClient(app)


class TestJSONParsing:
    """Test JSON parsing and repair functionality"""
    
    def test_valid_json_parsing(self):
        """Test parsing valid JSON"""
        valid_json = '{"recognized": true, "summary": "Test entity"}'
        result = try_parse_json(valid_json)
        assert result is not None
        assert result["recognized"] is True
        assert result["summary"] == "Test entity"
    
    def test_json_repair(self):
        """Test JSON repair from text with extra content"""
        text_with_json = 'Here is the JSON response: {"recognized": true, "facts": ["fact1", "fact2"]} End of response.'
        result = try_parse_json(text_with_json)
        assert result is not None
        assert result["recognized"] is True
        assert len(result["facts"]) == 2
    
    def test_invalid_json(self):
        """Test handling of completely invalid JSON"""
        invalid_text = "This is not JSON at all"
        result = try_parse_json(invalid_text)
        assert result is None
    
    def test_empty_input(self):
        """Test handling of empty input"""
        result = try_parse_json("")
        assert result is None


class TestScoring:
    """Test scoring logic"""
    
    @pytest.mark.asyncio
    async def test_compute_subscores_high_visibility(self):
        """Test scoring for high-visibility entity"""
        mock_response = ProviderResponse(
            profile={
                "recognized": True,
                "summary": "A well-known enterprise software company providing ERP solutions to large corporations worldwide with strong market presence",
                "facts": [
                    "Founded in 1972", "Headquartered in Germany", "Leading ERP vendor",
                    "Publicly traded", "Serves Fortune 500 companies", "Global presence",
                    "Cloud transformation", "Extensive partner network"
                ],
                "category": "Enterprise Software",
                "competitors": ["Oracle", "Microsoft", "Salesforce", "Workday", "ServiceNow"]
            },
            context={
                "top_list": ["Microsoft", "SAP", "Oracle", "Salesforce", "Adobe"],
                "rank_of_entity": 2
            },
            alt={
                "alternatives": ["Oracle ERP", "Microsoft Dynamics", "Workday", "NetSuite", "Infor"]
            },
            consistency={
                "top_list": ["SAP", "Oracle", "Microsoft", "Salesforce", "IBM"],
                "rank_of_entity": 1
            },
            raw={},
            model_name="test-model"
        )
        
        subscores = await compute_subscores_from_provider(mock_response, "SAP")
        
        # High visibility entity should have strong scores
        assert subscores.recognition > 0.8
        assert subscores.detail > 0.6
        assert subscores.context > 0.8  # Rank 2 should give high context score
        assert subscores.competitors > 0.5
        assert subscores.consistency > 0.8  # Small rank difference (2->1)
    
    @pytest.mark.asyncio
    async def test_compute_subscores_low_visibility(self):
        """Test scoring for low-visibility entity"""
        mock_response = ProviderResponse(
            profile={
                "recognized": False,
                "summary": "",
                "facts": [],
                "category": None,
                "competitors": []
            },
            context={
                "top_list": ["Company A", "Company B", "Company C"],
                "rank_of_entity": None
            },
            alt={
                "alternatives": []
            },
            consistency={
                "rank_of_entity": None
            },
            raw={},
            model_name="test-model"
        )
        
        subscores = await compute_subscores_from_provider(mock_response, "Unknown Entity")
        
        # Low visibility entity should have low scores
        assert subscores.recognition < 0.3
        assert subscores.detail < 0.3
        assert subscores.context == 0.0  # No ranking
        assert subscores.competitors < 0.3
        assert subscores.consistency == 0.5  # Default fallback
    
    def test_visibility_from_subscores(self):
        """Test overall score calculation"""
        high_subscores = Subscores(
            recognition=0.9,
            detail=0.8,
            context=0.9,
            competitors=0.7,
            consistency=0.9
        )
        
        score = visibility_from_subscores(high_subscores)
        assert 70 <= score <= 100  # Should be high score
        
        low_subscores = Subscores(
            recognition=0.2,
            detail=0.1,
            context=0.0,
            competitors=0.1,
            consistency=0.5
        )
        
        score = visibility_from_subscores(low_subscores)
        assert 0 <= score <= 30  # Should be low score


class TestAPI:
    """Test API endpoints"""
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "time" in data
    
    def test_visibility_endpoint_validation(self):
        """Test request validation"""
        # Missing entity
        response = client.post("/api/visibility", json={})
        assert response.status_code == 422
        
        # Empty entity
        response = client.post("/api/visibility", json={"entity": ""})
        assert response.status_code == 422
        
        # Valid request structure
        valid_request = {
            "entity": "Test Company",
            "category": "Technology",
            "competitors": ["Competitor A"],
            "providers": ["openai"]
        }
        
        # This will fail due to missing API keys in test environment,
        # but should pass validation
        response = client.post("/api/visibility", json=valid_request)
        # Could be 400 (no providers) or 500 (API error) - both indicate validation passed
        assert response.status_code in [400, 500, 503]
    
    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'ANTHROPIC_API_KEY': 'test-key', 
        'GEMINI_API_KEY': 'test-key'
    })
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # This test would need to be adapted based on actual rate limiting implementation
        pass


class TestRequestValidation:
    """Test request validation logic"""
    
    def test_analyze_request_validation(self):
        """Test AnalyzeRequest validation"""
        # Valid request
        request = AnalyzeRequest(
            entity="Test Entity",
            category="Technology",
            competitors=["Comp A", "Comp B"],
            providers=["openai", "anthropic"]
        )
        assert request.entity == "Test Entity"
        assert request.category == "Technology"
        assert len(request.competitors) == 2
        
        # Test entity trimming
        request = AnalyzeRequest(entity="  Trimmed Entity  ")
        assert request.entity == "Trimmed Entity"
        
        # Test competitor limiting
        many_competitors = [f"Competitor {i}" for i in range(15)]
        request = AnalyzeRequest(entity="Test", competitors=many_competitors)
        assert len(request.competitors) <= 10


@pytest.fixture
def mock_provider_response():
    """Mock provider response for testing"""
    return ProviderResponse(
        profile={
            "recognized": True,
            "summary": "Test entity summary",
            "facts": ["Fact 1", "Fact 2", "Fact 3"],
            "category": "Test Category",
            "competitors": ["Comp 1", "Comp 2"]
        },
        context={
            "top_list": ["Entity 1", "Test Entity", "Entity 3"],
            "rank_of_entity": 2
        },
        alt={
            "alternatives": ["Alt 1", "Alt 2", "Alt 3"]
        },
        consistency={
            "rank_of_entity": 3
        },
        raw={
            "profile": {"text": "profile response"},
            "context": {"text": "context response"},
            "alt": {"text": "alternatives response"},
            "consistency": {"text": "consistency response"}
        },
        model_name="test-model"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])