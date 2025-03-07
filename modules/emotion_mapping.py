"""
Emotion Mapping Module

This module maps semantic features and context into a weighted 
distribution of emotions.
"""

from typing import Dict, Any, List, Tuple

# Define the base emotions we'll be using
BASE_EMOTIONS = [
    "Happy",
    "Sad",
    "Angry",
    "Afraid",
    "Surprised",
    "Disgusted",
    "Curious",
    "Excited",
    "Calm"
]


def map_to_emotions(
    semantic_analysis: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, float]:
    """
    Map semantic analysis results and context into a distribution of emotions.
    
    Args:
        semantic_analysis: Results from the semantic analysis module
        context: Context determination from the context judgment module
        
    Returns:
        Dictionary mapping emotions to their intensity (0.0 to 1.0)
    """
    # Initialize all emotions with a base value
    emotions = {emotion: 0.05 for emotion in BASE_EMOTIONS}
    
    # Extract relevant data
    sentiment = semantic_analysis.get("sentiment", {})
    features = semantic_analysis.get("features", {})
    context_type = context.get("type", "unknown")
    
    # --- Base sentiment mapping ---
    
    # Map positive sentiment
    positive = sentiment.get("positive", 0)
    if positive > 0:
        emotions["Happy"] += positive * 0.5
        emotions["Excited"] += positive * 0.3
        emotions["Calm"] += positive * 0.2
    
    # Map negative sentiment
    negative = sentiment.get("negative", 0)
    if negative > 0:
        emotions["Sad"] += negative * 0.4
        emotions["Angry"] += negative * 0.3
        emotions["Disgusted"] += negative * 0.2
    
    # Map hostility
    hostility = sentiment.get("hostility", 0)
    if hostility > 0:
        # Context influences how hostility is interpreted
        if context_type == "friend":
            # Friends being hostile might be joking or playful
            emotions["Surprised"] += hostility * 0.4
            emotions["Curious"] += hostility * 0.3
            emotions["Excited"] += hostility * 0.2
        elif context_type == "enemy":
            # Enemies being hostile is threatening
            emotions["Afraid"] += hostility * 0.5
            emotions["Angry"] += hostility * 0.3
            emotions["Disgusted"] += hostility * 0.2
        else:
            # Neutral or unknown context - be cautious
            emotions["Surprised"] += hostility * 0.3
            emotions["Afraid"] += hostility * 0.3
            emotions["Curious"] += hostility * 0.2
    
    # --- Additional feature-based adjustments ---
    
    # Check for question indicators
    if features.get("is_question", False):
        emotions["Curious"] += 0.3
    
    # Look for specific keywords that might trigger emotions
    key_words = [word.lower() for word in features.get("key_words", [])]
    
    # Keyword-based emotion adjustments
    keyword_emotion_map = {
        "love": ("Happy", 0.3),
        "hate": ("Angry", 0.3),
        "kill": ("Afraid", 0.4) if context_type != "friend" else ("Surprised", 0.3),
        "die": ("Sad", 0.3),
        "amazing": ("Excited", 0.3),
        "terrible": ("Disgusted", 0.3),
        "surprise": ("Surprised", 0.4),
    }
    
    for word in key_words:
        if word in keyword_emotion_map:
            emotion, intensity = keyword_emotion_map[word]
            emotions[emotion] += intensity
    
    # --- Normalize the emotions ---
    
    # Ensure all emotions are between 0 and 1
    for emotion in emotions:
        emotions[emotion] = min(max(emotions[emotion], 0.0), 1.0)
    
    # Normalize to make the sum equal to 1.0
    total = sum(emotions.values())
    if total > 0:
        emotions = {k: v / total for k, v in emotions.items()}
    
    # Filter out emotions with very low values
    emotions = {k: v for k, v in emotions.items() if v >= 0.05}
    
    # Re-normalize after filtering
    total = sum(emotions.values())
    if total > 0:
        emotions = {k: v / total for k, v in emotions.items()}
    
    # Return the top emotions sorted by intensity
    sorted_emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_emotions


if __name__ == "__main__":
    # Example usage
    mock_semantic_analysis = {
        "sentiment": {
            "positive": 0.1,
            "negative": 0.2,
            "neutral": 0.7,
            "hostility": 0.6
        },
        "features": {
            "is_question": False,
            "key_words": ["kill", "you"]
        }
    }
    
    # Test with different contexts
    friend_context = {"type": "friend", "confidence": 0.9}
    enemy_context = {"type": "enemy", "confidence": 0.9}
    
    print("With friend context:")
    print(map_to_emotions(mock_semantic_analysis, friend_context))
    
    print("\nWith enemy context:")
    print(map_to_emotions(mock_semantic_analysis, enemy_context)) 