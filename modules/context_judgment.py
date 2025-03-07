"""
Context Judgment Module

This module determines the context of the conversation based on
the relationship between the speaker and the machine.
"""

from typing import Dict, Any, Literal, Optional, Union

# Define context types
ContextType = Literal["friend", "enemy", "neutral", "unknown"]


def determine_context(
    relationship: Optional[str] = None,
    history: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Determine the context of the conversation based on relationship and history.
    
    Args:
        relationship: Direct description of relationship (e.g., "friend", "enemy")
        history: Optional conversation history for context analysis
        metadata: Optional additional metadata about the speaker
        
    Returns:
        Dictionary containing context analysis results
    """
    # Initialize default context
    context: Dict[str, Union[str, float]] = {
        "type": "unknown",
        "confidence": 0.5,  # Default confidence
    }
    
    # If relationship is directly provided, use it as primary signal
    if relationship:
        relationship = relationship.lower()
        if relationship in ["friend", "ally", "friendly", "positive"]:
            context["type"] = "friend"
            context["confidence"] = 0.9
        elif relationship in ["enemy", "hostile", "negative", "adversary"]:
            context["type"] = "enemy"
            context["confidence"] = 0.9
        elif relationship in ["neutral", "stranger", "acquaintance"]:
            context["type"] = "neutral"
            context["confidence"] = 0.8
        else:
            # If relationship string doesn't match known categories
            context["type"] = "unknown"
            context["confidence"] = 0.5
    
    # If we have conversation history, analyze it (simplified version)
    if history and context["type"] == "unknown":
        # This would be a more complex analysis in a real system
        # For now, we'll use a simple rule-based approach
        friendly_indicators = 0
        hostile_indicators = 0
        
        # Count indicators in conversation history
        for entry in history.get("entries", []):
            if "sentiment" in entry:
                if entry["sentiment"].get("positive", 0) > 0.6:
                    friendly_indicators += 1
                if entry["sentiment"].get("negative", 0) > 0.6:
                    hostile_indicators += 1
                if entry["sentiment"].get("hostility", 0) > 0.3:
                    hostile_indicators += 2  # Weight hostility more heavily
        
        # Determine context from indicators
        if friendly_indicators > hostile_indicators * 2:
            context["type"] = "friend"
            context["confidence"] = 0.7
        elif hostile_indicators > friendly_indicators:
            context["type"] = "enemy"
            context["confidence"] = 0.7
        else:
            context["type"] = "neutral"
            context["confidence"] = 0.6
    
    # Add any relevant metadata
    if metadata:
        # Add custom flags that might be helpful for emotion mapping
        if metadata.get("known_person") is True:
            context["known_person"] = True
        if "relationship_duration" in metadata:
            context["relationship_duration"] = metadata["relationship_duration"]
    
    return context


if __name__ == "__main__":
    # Example usage
    print(determine_context(relationship="friend"))
    print(determine_context(relationship="enemy"))
    
    # Example with history
    mock_history = {
        "entries": [
            {"sentiment": {"positive": 0.8, "negative": 0.1, "hostility": 0.0}},
            {"sentiment": {"positive": 0.7, "negative": 0.2, "hostility": 0.1}}
        ]
    }
    print(determine_context(history=mock_history)) 