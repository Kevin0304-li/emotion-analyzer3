"""
Emotion Mapping Module

This module maps semantic features and context into a weighted 
distribution of emotions.
"""

from typing import Dict, Any, List, Tuple

# Define the base emotions we'll be using
BASE_EMOTIONS = [
    "Happy",      # Joy, pleasure, contentment
    "Sad",        # Grief, sorrow, unhappiness
    "Angry",      # Rage, annoyance, frustration
    "Afraid",     # Fear, worry, anxiety
    "Surprised",  # Astonishment, shock
    "Disgusted",  # Revulsion, contempt
    "Curious",    # Interest, wonder, inquisitiveness
    "Excited",    # Enthusiasm, eagerness, anticipation
    "Calm",       # Serenity, tranquility, peace
    "Confused",   # Bewilderment, uncertainty, doubt
    "Amused",     # Entertainment, humor appreciation
    "Proud",      # Self-respect, dignity, satisfaction
    "Grateful",   # Thankful, appreciative
    "Loving",     # Affection, care, attachment
    "Nervous"     # Apprehension, unease, restlessness
]

# Pattern-based emotion triggers (regexes would be better but keeping it simple)
EMOTION_PATTERNS = {
    "Happy": ["thank you", "grateful", "appreciate", "congratulations", "well done", "good job"],
    "Sad": ["miss you", "lost", "alone", "sorry for", "condolences", "sympathy"],
    "Afraid": ["help me", "save me", "danger", "scared", "afraid of", "worried about"],
    "Curious": ["how do", "why does", "what if", "tell me about", "explain", "curious about"],
    "Confused": ["don't understand", "confused about", "unclear", "not sure", "perplexed"],
    "Amused": ["funny", "joke", "lol", "haha", "lmao", "hilarious"],
    "Nervous": ["nervous about", "anxious", "worried", "stress", "concerned"]
}

# Word combos that override individual word scores
EMOTION_PHRASES = {
    "miss you": ("Sad", 0.7),
    "hate you": ("Afraid", 0.4, "enemy"),  # With context-specific mapping
    "hate you": ("Angry", 0.7, "friend"),
    "love you": ("Happy", 0.8),
    "good job": ("Proud", 0.6),
    "thank you": ("Grateful", 0.7),
    "so excited": ("Excited", 0.8),
    "feel sad": ("Sad", 0.7),
    "just kidding": ("Amused", 0.6),
    "very happy": ("Happy", 0.8),
    "totally confused": ("Confused", 0.8)
}

# Context-specific emotion weights for different relationship types
CONTEXT_WEIGHTS = {
    "friend": {
        "Happy": 1.3,
        "Excited": 1.2, 
        "Amused": 1.4,
        "Loving": 1.5,
        "Afraid": 0.6,  # Less afraid with friends
        "Angry": 0.7    # Less angry with friends
    },
    "enemy": {
        "Afraid": 1.5,
        "Angry": 1.3,
        "Disgusted": 1.2,
        "Happy": 0.6,
        "Loving": 0.3,
        "Calm": 0.5
    },
    "neutral": {
        "Curious": 1.3,
        "Confused": 1.1,
        "Nervous": 1.2
    }
}


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
    context_confidence = context.get("confidence", 0.5)
    
    # --- Process the input text for context-specific patterns ---
    
    # Check for emotion phrases in the original text
    text = semantic_analysis.get("input_text", "")
    if text:
        text_lower = text.lower()
        
        # Check for complete phrases
        for phrase, (emotion, intensity, *ctx) in EMOTION_PHRASES.items():
            if phrase in text_lower:
                # If the phrase has a context restriction, check it
                if len(ctx) > 0 and ctx[0] != context_type:
                    continue
                emotions[emotion] += intensity * context_confidence
        
        # Check for emotion patterns
        for emotion, patterns in EMOTION_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    emotions[emotion] += 0.4 * context_confidence
    
    # --- Base sentiment mapping ---
    
    # Map positive sentiment with more nuance
    positive = sentiment.get("positive", 0)
    if positive > 0:
        emotions["Happy"] += positive * 0.5
        emotions["Excited"] += positive * 0.25
        emotions["Calm"] += positive * 0.15
        emotions["Loving"] += positive * 0.2
        emotions["Proud"] += positive * 0.15
        emotions["Grateful"] += positive * 0.15
    
    # Map negative sentiment with more nuance
    negative = sentiment.get("negative", 0)
    if negative > 0:
        emotions["Sad"] += negative * 0.4
        emotions["Angry"] += negative * 0.3
        emotions["Disgusted"] += negative * 0.2
        emotions["Nervous"] += negative * 0.2
    
    # Map hostility with more nuanced context handling
    hostility = sentiment.get("hostility", 0)
    if hostility > 0:
        # Context influences how hostility is interpreted
        if context_type == "friend":
            # Friends being hostile might be joking or playful
            context_factor = max(0.2, 1.0 - context_confidence)  # Less confident = more cautious
            emotions["Surprised"] += hostility * 0.4
            emotions["Curious"] += hostility * 0.3
            emotions["Excited"] += hostility * 0.2
            emotions["Amused"] += hostility * 0.3  # Friends might be joking
            
            # Even with friends, high hostility should cause some concern
            if hostility > 0.7:
                emotions["Confused"] += hostility * 0.3
                emotions["Afraid"] += hostility * context_factor
        
        elif context_type == "enemy":
            # Enemies being hostile is threatening
            emotions["Afraid"] += hostility * 0.5
            emotions["Angry"] += hostility * 0.3
            emotions["Disgusted"] += hostility * 0.2
            
            # High confidence in enemy status increases negative emotions
            if context_confidence > 0.8:
                emotions["Nervous"] += hostility * 0.4
        
        else:
            # Neutral or unknown context - be cautious
            emotions["Surprised"] += hostility * 0.3
            emotions["Afraid"] += hostility * 0.3
            emotions["Curious"] += hostility * 0.2
            emotions["Confused"] += hostility * 0.2
    
    # --- Handle emotional intensity ---
    
    # Overall emotional intensity affects the strength of emotions
    intensity = sentiment.get("intensity", 0.5)
    for emotion in emotions:
        emotions[emotion] *= (0.5 + intensity/2)  # Scale emotions by intensity
    
    # --- Additional feature-based adjustments ---
    
    # Check for question indicators - questions increase curiosity
    if features.get("is_question", False):
        emotions["Curious"] += 0.3
        emotions["Confused"] += 0.1
    
    # Check for exclamations - increase emotional intensity
    if features.get("exclamatory", False):
        # Exclamations amplify the strongest emotions
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        for emotion, _ in top_emotions:
            emotions[emotion] *= 1.3
    
    # Check for imperative sentences - commands affect emotions differently
    if features.get("imperative", False):
        if context_type == "friend":
            emotions["Curious"] += 0.2
        elif context_type == "enemy":
            emotions["Angry"] += 0.2
            emotions["Disgusted"] += 0.1
    
    # Look for specific keywords that might trigger emotions
    key_words = [word.lower() for word in features.get("key_words", [])]
    
    # Expanded keyword-based emotion adjustments
    keyword_emotion_map = {
        # Positive emotional triggers
        "love": [("Happy", 0.3), ("Loving", 0.4)],
        "like": [("Happy", 0.2), ("Calm", 0.1)],
        "happy": [("Happy", 0.4), ("Excited", 0.2)],
        "good": [("Happy", 0.2), ("Calm", 0.1)],
        "great": [("Happy", 0.3), ("Excited", 0.2)],
        "thank": [("Happy", 0.2), ("Grateful", 0.4)],
        "appreciate": [("Grateful", 0.5), ("Happy", 0.2)],
        "excited": [("Excited", 0.5), ("Happy", 0.2)],
        "glad": [("Happy", 0.3), ("Grateful", 0.1)],
        "wonderful": [("Happy", 0.4), ("Excited", 0.2)],
        
        # Negative emotional triggers
        "hate": [("Angry", 0.3), ("Disgusted", 0.3)],
        "angry": [("Angry", 0.5), ("Disgusted", 0.1)],
        "sad": [("Sad", 0.5), ("Calm", 0.1)],
        "upset": [("Sad", 0.3), ("Angry", 0.2)],
        "annoyed": [("Angry", 0.3), ("Disgusted", 0.1)],
        "worried": [("Afraid", 0.3), ("Nervous", 0.3)],
        
        # Threat words - context-dependent interpretation
        "kill": lambda ctx: [("Afraid", 0.5), ("Angry", 0.3)] if ctx != "friend" 
                else [("Surprised", 0.4), ("Confused", 0.2), ("Amused", 0.2)],
        "hurt": lambda ctx: [("Afraid", 0.4), ("Sad", 0.2)] if ctx != "friend"
                else [("Surprised", 0.3), ("Confused", 0.2)],
        "die": lambda ctx: [("Sad", 0.4), ("Afraid", 0.3)] if ctx != "friend"
                else [("Surprised", 0.3), ("Sad", 0.2)],
        "attack": lambda ctx: [("Afraid", 0.4), ("Angry", 0.3)] if ctx != "friend"
                else [("Surprised", 0.3), ("Confused", 0.2)],
        
        # Question/curiosity triggers
        "why": [("Curious", 0.4), ("Confused", 0.2)],
        "how": [("Curious", 0.4), ("Confused", 0.1)],
        "what": [("Curious", 0.3), ("Confused", 0.1)],
        "wonder": [("Curious", 0.5), ("Excited", 0.1)],
        
        # Surprise triggers
        "surprise": [("Surprised", 0.5), ("Excited", 0.2)],
        "unexpected": [("Surprised", 0.4), ("Confused", 0.2)],
        "amazing": [("Surprised", 0.3), ("Excited", 0.3), ("Happy", 0.2)],
        "shocking": [("Surprised", 0.5), ("Confused", 0.2)],
        
        # Confusion triggers
        "confused": [("Confused", 0.5), ("Curious", 0.2)],
        "understand": [("Confused", 0.2), ("Curious", 0.2)]
    }
    
    # Process keywords for emotional impact
    for word in key_words:
        if word in keyword_emotion_map:
            # Handle context-dependent mappings
            emotion_impacts = keyword_emotion_map[word]
            if callable(emotion_impacts):
                emotion_impacts = emotion_impacts(context_type)
                
            # Apply all emotion impacts
            for emotion, intensity in emotion_impacts:
                emotions[emotion] += intensity
    
    # --- Apply context-specific weighting ---
    
    # Adjust emotions based on relationship context
    if context_type in CONTEXT_WEIGHTS:
        for emotion, weight in CONTEXT_WEIGHTS[context_type].items():
            if emotion in emotions:
                emotions[emotion] *= weight * context_confidence
    
    # --- Normalize the emotions ---
    
    # Ensure all emotions are between 0 and 1
    for emotion in emotions:
        emotions[emotion] = min(max(emotions[emotion], 0.0), 1.0)
    
    # Normalize to make the sum equal to 1.0
    total = sum(emotions.values())
    if total > 0:
        emotions = {k: v / total for k, v in emotions.items()}
    
    # Filter out emotions with very low values (increased threshold slightly)
    emotions = {k: v for k, v in emotions.items() if v >= 0.06}
    
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
        "input_text": "I'm going to kill you",
        "sentiment": {
            "positive": 0.1,
            "negative": 0.2,
            "neutral": 0.7,
            "hostility": 0.6,
            "intensity": 0.7
        },
        "features": {
            "is_question": False,
            "key_words": ["kill", "you"],
            "imperative": False,
            "exclamatory": False
        }
    }
    
    # Test with different contexts
    friend_context = {"type": "friend", "confidence": 0.9}
    enemy_context = {"type": "enemy", "confidence": 0.9}
    
    print("With friend context:")
    print(map_to_emotions(mock_semantic_analysis, friend_context))
    
    print("\nWith enemy context:")
    print(map_to_emotions(mock_semantic_analysis, enemy_context))
    
    # Test with a positive example
    positive_analysis = {
        "input_text": "I really love what you've done, it's amazing work!",
        "sentiment": {
            "positive": 0.7,
            "negative": 0.0,
            "neutral": 0.3,
            "hostility": 0.0,
            "intensity": 0.6
        },
        "features": {
            "is_question": False,
            "key_words": ["love", "amazing", "work"],
            "imperative": False,
            "exclamatory": True
        }
    }
    
    print("\nPositive example:")
    print(map_to_emotions(positive_analysis, friend_context)) 