"""
Tests for the Semantic Analysis module.
"""

import unittest
import sys
import os

# Add the parent directory to the sys path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.semantic_analysis import analyze_text, analyze_sentiment


class TestSemanticAnalysis(unittest.TestCase):
    
    def test_analyze_text_basic(self):
        """Test basic text analysis functionality."""
        text = "I am happy today."
        result = analyze_text(text)
        
        # Check that the result has the expected structure
        self.assertIn("features", result)
        self.assertIn("sentiment", result)
        
        # Check that features contains expected keys
        features = result["features"]
        self.assertIn("key_words", features)
        self.assertIn("word_count", features)
        
        # Check that sentiment analysis was performed
        sentiment = result["sentiment"]
        self.assertIn("positive", sentiment)
        self.assertIn("negative", sentiment)
        self.assertIn("neutral", sentiment)
        
        # For this positive text, positive score should be > negative
        self.assertGreater(sentiment["positive"], sentiment["negative"])
    
    def test_analyze_text_negative(self):
        """Test text analysis with negative content."""
        text = "I hate this terrible day."
        result = analyze_text(text)
        
        # Check sentiment scores for negative content
        sentiment = result["sentiment"]
        self.assertGreater(sentiment["negative"], sentiment["positive"])
    
    def test_analyze_text_hostile(self):
        """Test text analysis with hostile content."""
        text = "I'm going to kill you."
        result = analyze_text(text)
        
        # Check hostility score
        sentiment = result["sentiment"]
        self.assertIn("hostility", sentiment)
        self.assertGreater(sentiment["hostility"], 0)
    
    def test_analyze_text_empty(self):
        """Test handling of empty text."""
        text = ""
        result = analyze_text(text)
        
        # Should still return a valid structure
        self.assertIn("features", result)
        self.assertIn("sentiment", result)
    
    def test_analyze_sentiment_direct(self):
        """Test sentiment analysis function directly."""
        # Since analyze_sentiment expects a spaCy doc, we need to create one first
        # This test depends on spaCy being properly installed
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp("I love this! It's amazing and wonderful.")
            sentiment = analyze_sentiment(doc)
            
            # Check that positive sentiment is detected
            self.assertGreater(sentiment["positive"], 0)
            
            # For this text, positive should be higher than negative
            self.assertGreater(sentiment["positive"], sentiment["negative"])
            
        except OSError:
            # Skip this test if spaCy model isn't installed
            self.skipTest("Spacy model not available - skipping test")


if __name__ == "__main__":
    unittest.main() 