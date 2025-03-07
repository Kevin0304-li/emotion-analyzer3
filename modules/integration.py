"""
Integration Module

This module integrates the outputs from the semantic analysis, 
context judgment, and emotion mapping modules to produce the 
final emotional response.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime

from modules.semantic_analysis import analyze_text
from modules.context_judgment import determine_context
from modules.emotion_mapping import map_to_emotions
from modules.context_learning import (
    save_interaction, 
    adjust_emotions_with_learning,
    get_similar_emotional_responses
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emotion_calculator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("emotion_calculator")


def process_input(
    text: str,
    relationship: Optional[str] = None,
    history: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    use_learning: bool = True,
    feedback: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Process user input to generate an emotional response.
    
    Args:
        text: Input text to analyze
        relationship: Optional relationship context (e.g., "friend", "enemy")
        history: Optional conversation history
        metadata: Optional additional metadata
        debug: Whether to include debug information in the response
        use_learning: Whether to use context learning to adjust emotions
        feedback: Optional user feedback to incorporate into learning
        
    Returns:
        Dictionary containing the emotional response and optional debug info
    """
    logger.info(f"Processing input: '{text}' with relationship={relationship}")
    
    # Step 1: Perform semantic analysis
    semantic_analysis = analyze_text(text)
    
    # Add the original text to semantic analysis for pattern matching
    semantic_analysis["input_text"] = text
    
    # Step 2: Determine context
    context = determine_context(
        relationship=relationship,
        history=history,
        metadata=metadata
    )
    
    # Log the determined context
    logger.info(f"Determined context: {context.get('type', 'unknown')} with confidence {context.get('confidence', 0):.2f}")
    
    # Step 3: Map to emotions
    emotions = map_to_emotions(semantic_analysis, context)
    
    # Step 4: Apply context learning if enabled
    if use_learning:
        # Check for similar past interactions
        similar_interactions = get_similar_emotional_responses(
            text, 
            context.get("type", "unknown"),
            limit=3
        )
        
        # Apply learned adjustments to emotions
        original_emotions = emotions.copy()
        emotions = adjust_emotions_with_learning(
            text, 
            context.get("type", "unknown"), 
            emotions
        )
        
        # Log learning adjustments
        if original_emotions != emotions:
            logger.info("Emotions adjusted based on learning")
            
        # Add similar responses to the result if in debug mode
        if debug and similar_interactions:
            semantic_analysis["similar_responses"] = similar_interactions
    
    # Log the top emotions
    top_emotions = list(emotions.items())[:3] if emotions else []
    logger.info(f"Top emotions: {format_emotions({k: v for k, v in top_emotions})}")
    
    # Prepare the response
    response = {
        "emotions": emotions,
        "dominant_emotion": next(iter(emotions), None),
        "input_text": text,
        "context": context.get("type", "unknown"),
        "context_confidence": context.get("confidence", 0.0),
        "timestamp": datetime.now().isoformat()
    }
    
    # Add debug information if requested
    if debug:
        response["debug"] = {
            "semantic_analysis": semantic_analysis,
            "context_full": context,
            "learning_applied": use_learning
        }
    
    # Save interaction for learning if enabled
    if use_learning:
        save_interaction(
            text=text,
            emotions=emotions,
            context=context.get("type", "unknown"),
            feedback=feedback
        )
    
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


def provide_feedback(
    response_id: str,
    feedback_emotions: Dict[str, float]
) -> bool:
    """
    Provide feedback on an emotional response to improve future accuracy.
    
    Args:
        response_id: Timestamp or ID of the response to provide feedback for
        feedback_emotions: The correct emotions as feedback
        
    Returns:
        Whether feedback was successfully applied
    """
    # Not implemented in this version, but could be used to adjust learning
    logger.info(f"Feedback received for response {response_id}")
    return True


def analyze_response_quality(
    expected_emotions: Dict[str, float], 
    actual_emotions: Dict[str, float]
) -> Dict[str, Any]:
    """
    Analyze the quality of the emotional response compared to expected emotions.
    
    Args:
        expected_emotions: Dictionary of expected emotions
        actual_emotions: Dictionary of actual emotions produced
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        "matched_emotions": 0,
        "missing_emotions": [],
        "unexpected_emotions": [],
        "accuracy": 0.0
    }
    
    # Count matched emotions
    for emotion in expected_emotions:
        if emotion in actual_emotions:
            metrics["matched_emotions"] += 1
        else:
            metrics["missing_emotions"].append(emotion)
    
    # Find unexpected emotions
    for emotion in actual_emotions:
        if emotion not in expected_emotions:
            metrics["unexpected_emotions"].append(emotion)
    
    # Calculate basic accuracy
    if expected_emotions:
        metrics["accuracy"] = metrics["matched_emotions"] / len(expected_emotions)
    
    return metrics


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
    
    # Test a positive example
    positive_text = "Thank you so much for your help, you're amazing!"
    response_positive = process_input(positive_text, relationship="friend")
    print(f"\nInput: '{positive_text}'")
    print(f"Emotional Response: {format_emotions(response_positive['emotions'])}") 