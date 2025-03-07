"""
Integration Module

This module integrates the outputs from the semantic analysis, 
context judgment, and emotion mapping modules to produce the 
final emotional response.
"""

import json
from typing import Dict, Any, Optional, Tuple, List

from modules.semantic_analysis import analyze_text
from modules.context_judgment import determine_context
from modules.emotion_mapping import map_to_emotions


def process_input(
    text: str,
    relationship: Optional[str] = None,
    history: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Process user input to generate an emotional response.
    
    Args:
        text: Input text to analyze
        relationship: Optional relationship context (e.g., "friend", "enemy")
        history: Optional conversation history
        metadata: Optional additional metadata
        debug: Whether to include debug information in the response
        
    Returns:
        Dictionary containing the emotional response and optional debug info
    """
    # Step 1: Perform semantic analysis
    semantic_analysis = analyze_text(text)
    
    # Step 2: Determine context
    context = determine_context(
        relationship=relationship,
        history=history,
        metadata=metadata
    )
    
    # Step 3: Map to emotions
    emotions = map_to_emotions(semantic_analysis, context)
    
    # Prepare the response
    response = {
        "emotions": emotions,
        "dominant_emotion": next(iter(emotions), None),
        "input_text": text,
        "context": context.get("type", "unknown"),
        "context_confidence": context.get("confidence", 0.0)
    }
    
    # Add debug information if requested
    if debug:
        response["debug"] = {
            "semantic_analysis": semantic_analysis,
            "context_full": context
        }
    
    return response


def format_emotions(emotions: Dict[str, float], decimal_places: int = 2) -> str:
    """
    Format emotions dictionary as a readable string with percentages.
    
    Args:
        emotions: Dictionary of emotions and their intensities
        decimal_places: Number of decimal places to show in percentages
        
    Returns:
        Formatted string of emotions with percentages
    """
    if not emotions:
        return "No emotions detected"
    
    # Format each emotion as a percentage
    formatted_emotions = [
        f"{emotion} {value * 100:.{decimal_places}f}%" 
        for emotion, value in emotions.items()
    ]
    
    return ", ".join(formatted_emotions)


def export_to_json(response: Dict[str, Any], filepath: str) -> None:
    """
    Export the response to a JSON file.
    
    Args:
        response: The response dictionary to export
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(response, f, indent=2)


if __name__ == "__main__":
    # Example usage
    test_text = "I'm going to kill you"
    
    print("\nTesting with friend relationship:")
    response_friend = process_input(test_text, relationship="friend", debug=True)
    print(f"Input: '{test_text}'")
    print(f"Context: {response_friend['context']} (confidence: {response_friend['context_confidence']:.2f})")
    print(f"Emotional Response: {format_emotions(response_friend['emotions'])}")
    
    print("\nTesting with enemy relationship:")
    response_enemy = process_input(test_text, relationship="enemy", debug=True)
    print(f"Input: '{test_text}'")
    print(f"Context: {response_enemy['context']} (confidence: {response_enemy['context_confidence']:.2f})")
    print(f"Emotional Response: {format_emotions(response_enemy['emotions'])}") 