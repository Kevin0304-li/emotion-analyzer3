"""
Tests for the Context Judgment module.
"""

import unittest
import sys
import os

# Add the parent directory to the sys path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.context_judgment import determine_context


class TestContextJudgment(unittest.TestCase):
    
    def test_determine_context_direct_relationship(self):
        """Test context determination with direct relationship specification."""
        # Test friend context
        result = determine_context(relationship="friend")
        self.assertEqual(result["type"], "friend")
        self.assertGreater(result["confidence"], 0.8)
        
        # Test enemy context
        result = determine_context(relationship="enemy")
        self.assertEqual(result["type"], "enemy")
        self.assertGreater(result["confidence"], 0.8)
        
        # Test neutral context
        result = determine_context(relationship="neutral")
        self.assertEqual(result["type"], "neutral")
        self.assertGreater(result["confidence"], 0.7)
    
    def test_determine_context_alternative_terms(self):
        """Test context determination with alternative relationship terms."""
        # Test alternative friend terms
        for term in ["ally", "friendly", "positive"]:
            result = determine_context(relationship=term)
            self.assertEqual(result["type"], "friend")
        
        # Test alternative enemy terms
        for term in ["hostile", "adversary", "negative"]:
            result = determine_context(relationship=term)
            self.assertEqual(result["type"], "enemy")
    
    def test_determine_context_unknown(self):
        """Test context determination with unknown relationship."""
        result = determine_context(relationship="something else")
        self.assertEqual(result["type"], "unknown")
        
        # Test with no inputs
        result = determine_context()
        self.assertEqual(result["type"], "unknown")
    
    def test_determine_context_from_history(self):
        """Test context determination based on conversation history."""
        # Mock history with positive sentiment
        positive_history = {
            "entries": [
                {"sentiment": {"positive": 0.8, "negative": 0.1, "hostility": 0.0}},
                {"sentiment": {"positive": 0.7, "negative": 0.2, "hostility": 0.1}}
            ]
        }
        
        result = determine_context(history=positive_history)
        self.assertEqual(result["type"], "friend")
        
        # Mock history with negative sentiment
        negative_history = {
            "entries": [
                {"sentiment": {"positive": 0.1, "negative": 0.8, "hostility": 0.5}},
                {"sentiment": {"positive": 0.2, "negative": 0.7, "hostility": 0.4}}
            ]
        }
        
        result = determine_context(history=negative_history)
        self.assertEqual(result["type"], "enemy")
    
    def test_determine_context_with_metadata(self):
        """Test context determination with additional metadata."""
        metadata = {
            "known_person": True,
            "relationship_duration": "long"
        }
        
        result = determine_context(relationship="friend", metadata=metadata)
        self.assertEqual(result["type"], "friend")
        self.assertTrue(result.get("known_person"))
        self.assertEqual(result.get("relationship_duration"), "long")


if __name__ == "__main__":
    unittest.main() 