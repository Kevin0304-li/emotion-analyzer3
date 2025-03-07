"""
Tests for the Emotion Mapping module.
"""

import unittest
import sys
import os

# Add the parent directory to the sys path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.emotion_mapping import map_to_emotions, BASE_EMOTIONS


class TestEmotionMapping(unittest.TestCase):
    
    def test_map_to_emotions_basic(self):
        """Test basic emotion mapping functionality."""
        # Create a mock semantic analysis result
        mock_semantic_analysis = {
            "sentiment": {
                "positive": 0.8,
                "negative": 0.1,
                "neutral": 0.1,
                "hostility": 0.0
            },
            "features": {
                "is_question": False,
                "key_words": ["love", "happy"]
            }
        }
        
        # Create a mock context
        mock_context = {
            "type": "friend",
            "confidence": 0.9
        }
        
        # Get emotion mapping
        emotions = map_to_emotions(mock_semantic_analysis, mock_context)
        
        # Verify result structure
        self.assertIsInstance(emotions, dict)
        self.assertTrue(len(emotions) > 0)
        
        # For this positive input, "Happy" should be one of the top emotions
        self.assertIn("Happy", emotions)
        
        # Sum of all emotion values should be approximately 1.0
        self.assertAlmostEqual(sum(emotions.values()), 1.0, places=6)
    
    def test_map_to_emotions_hostile_friend(self):
        """Test emotion mapping with hostile content from a friend."""
        # Create a mock semantic analysis with hostile content
        mock_semantic_analysis = {
            "sentiment": {
                "positive": 0.1,
                "negative": 0.3,
                "neutral": 0.6,
                "hostility": 0.8
            },
            "features": {
                "is_question": False,
                "key_words": ["kill", "you"]
            }
        }
        
        # Friend context
        friend_context = {
            "type": "friend",
            "confidence": 0.9
        }
        
        # Get emotion mapping for friend context
        emotions_friend = map_to_emotions(mock_semantic_analysis, friend_context)
        
        # For hostile content from a friend, we expect surprise/curiosity rather than fear
        self.assertIn("Surprised", emotions_friend)
        if "Afraid" in emotions_friend and "Surprised" in emotions_friend:
            self.assertGreater(emotions_friend["Surprised"], emotions_friend["Afraid"])
    
    def test_map_to_emotions_hostile_enemy(self):
        """Test emotion mapping with hostile content from an enemy."""
        # Create a mock semantic analysis with hostile content
        mock_semantic_analysis = {
            "sentiment": {
                "positive": 0.1,
                "negative": 0.3,
                "neutral": 0.6,
                "hostility": 0.8
            },
            "features": {
                "is_question": False,
                "key_words": ["kill", "you"]
            }
        }
        
        # Enemy context
        enemy_context = {
            "type": "enemy",
            "confidence": 0.9
        }
        
        # Get emotion mapping for enemy context
        emotions_enemy = map_to_emotions(mock_semantic_analysis, enemy_context)
        
        # For hostile content from an enemy, we expect fear to be prominent
        self.assertIn("Afraid", emotions_enemy)
        if "Afraid" in emotions_enemy:
            # Fear should be one of the top emotions
            top_emotions = list(emotions_enemy.keys())[:3]
            self.assertIn("Afraid", top_emotions)
    
    def test_map_to_emotions_question(self):
        """Test emotion mapping with a question."""
        # Create a mock semantic analysis with a question
        mock_semantic_analysis = {
            "sentiment": {
                "positive": 0.2,
                "negative": 0.1,
                "neutral": 0.7,
                "hostility": 0.0
            },
            "features": {
                "is_question": True,
                "key_words": ["why", "how"]
            }
        }
        
        # Neutral context
        context = {
            "type": "neutral",
            "confidence": 0.8
        }
        
        # Get emotion mapping
        emotions = map_to_emotions(mock_semantic_analysis, context)
        
        # For a question, "Curious" should be prominent
        self.assertIn("Curious", emotions)
        
        # Curious should be one of the top emotions
        top_emotions = list(emotions.keys())[:3]
        self.assertIn("Curious", top_emotions)
    
    def test_map_to_emotions_normalization(self):
        """Test that emotion values are properly normalized."""
        # Create minimal mock inputs
        mock_semantic_analysis = {
            "sentiment": {
                "positive": 0.5,
                "negative": 0.5,
                "neutral": 0.0,
                "hostility": 0.0
            },
            "features": {}
        }
        
        mock_context = {
            "type": "neutral",
            "confidence": 0.5
        }
        
        # Get emotion mapping
        emotions = map_to_emotions(mock_semantic_analysis, mock_context)
        
        # Verify that the sum of all emotion values is 1.0
        self.assertAlmostEqual(sum(emotions.values()), 1.0, places=6)
        
        # All values should be between 0 and 1
        for value in emotions.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main() 