"""
Emotion Calculator

Main entry point for the emotion calculator application.
"""

import argparse
import json
import sys
from typing import Dict, Any, Optional

from modules.integration import (
    process_input, 
    format_emotions, 
    export_to_json,
    provide_feedback
)
from modules.context_learning import get_similar_emotional_responses


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Emotion Calculator")
    parser.add_argument("--text", "-t", type=str, help="Input text to analyze")
    parser.add_argument("--relationship", "-r", type=str, 
                        help="Relationship context (friend, enemy, neutral)")
    parser.add_argument("--debug", "-d", action="store_true", 
                        help="Include debug information in the output")
    parser.add_argument("--output", "-o", type=str, 
                        help="Save output to JSON file at the specified path")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--no-learning", action="store_true",
                        help="Disable context learning from past interactions")
    parser.add_argument("--similar", "-s", action="store_true",
                        help="Show similar past responses")
    
    return parser.parse_args()


def interactive_mode(use_learning=True):
    """Run the emotion calculator in interactive mode."""
    print("=== Emotion Calculator Interactive Mode ===")
    print("Type 'exit' to quit.")
    print("Other commands:")
    print("  'similar' - Show similar past responses for last input")
    print("  'feedback' - Provide feedback on the last response")
    
    # Keep track of conversation history for context
    history = {"entries": []}
    relationship = None
    last_response = None
    last_text = None
    
    # First, get the relationship context
    while relationship is None:
        rel_input = input("\nWhat is your relationship to the machine? (friend/enemy/neutral): ").strip().lower()
        if rel_input in ["friend", "enemy", "neutral"]:
            relationship = rel_input
        elif rel_input == "exit":
            return
        else:
            print("Please enter 'friend', 'enemy', or 'neutral'.")
    
    print(f"\nRelationship set to: {relationship}")
    print("Learning mode: " + ("ON" if use_learning else "OFF"))
    print("\nEnter text to analyze emotions. Type 'exit' to quit.")
    
    while True:
        # Get user input
        text = input("\nYou: ").strip()
        
        # Check for commands
        if text.lower() == "exit":
            break
        
        if text.lower() == "similar" and last_text:
            similar_responses = get_similar_emotional_responses(last_text, relationship)
            if similar_responses:
                print("\nSimilar past responses:")
                for i, resp in enumerate(similar_responses, 1):
                    print(f"{i}. '{resp.get('input_text', '')}' ({resp.get('context', 'unknown')})")
                    print(f"   Emotions: {format_emotions(resp.get('emotions', {}))}")
            else:
                print("\nNo similar past responses found.")
            continue
            
        if text.lower() == "feedback" and last_response:
            print("\nProvide feedback on the last emotional response.")
            print("Enter emotions and percentages, e.g.: Happy 50, Surprised 30, Curious 20")
            feedback_text = input("Feedback: ").strip()
            
            if feedback_text:
                try:
                    # Parse feedback into a dictionary
                    feedback = {}
                    for item in feedback_text.split(","):
                        emotion, value = item.strip().split()
                        feedback[emotion] = float(value) / 100.0
                    
                    # Normalize feedback
                    total = sum(feedback.values())
                    if total > 0:
                        feedback = {k: v/total for k, v in feedback.items()}
                    
                    # Provide feedback
                    response_id = last_response.get("timestamp", "unknown")
                    if provide_feedback(response_id, feedback):
                        print("Feedback recorded. Thank you!")
                except Exception as e:
                    print(f"Error processing feedback: {e}")
                    print("Please use the format: Emotion Percentage, Emotion Percentage, ...")
            continue
        
        if not text:
            continue
            
        # Save the current text for potential "similar" command later
        last_text = text
        
        # Process the input
        response = process_input(
            text=text,
            relationship=relationship,
            history=history if history["entries"] else None,
            use_learning=use_learning
        )
        
        # Save the response for feedback
        last_response = response
        
        # Update history
        history["entries"].append({
            "text": text,
            "sentiment": response.get("debug", {}).get("semantic_analysis", {}).get("sentiment", {})
        })
        
        # Display the emotional response
        print("\nMachine's emotional response:")
        print(format_emotions(response["emotions"]))
        
        # Display the dominant emotion with its intensity
        if response["emotions"]:
            dominant_emotion, intensity = next(iter(response["emotions"].items()))
            print(f"Dominant emotion: {dominant_emotion} ({intensity*100:.1f}%)")


def main():
    """Main function to run the emotion calculator."""
    args = parse_arguments()
    
    # Interactive mode
    if args.interactive:
        interactive_mode(use_learning=not args.no_learning)
        return
    
    # Command-line mode
    if not args.text:
        print("Please provide text to analyze using the --text parameter.")
        print("Or use --interactive for interactive mode.")
        return
    
    # Process the input
    response = process_input(
        text=args.text,
        relationship=args.relationship,
        debug=args.debug,
        use_learning=not args.no_learning
    )
    
    # Display the results
    print("\nInput text:", args.text)
    print("Context:", response["context"])
    print("\nEmotional Response:")
    print(format_emotions(response["emotions"]))
    
    # Display similar responses if requested
    if args.similar:
        similar_responses = get_similar_emotional_responses(
            args.text, 
            response["context"]
        )
        if similar_responses:
            print("\nSimilar past responses:")
            for i, resp in enumerate(similar_responses, 1):
                print(f"{i}. '{resp.get('input_text', '')}' ({resp.get('context', 'unknown')})")
                print(f"   Emotions: {format_emotions(resp.get('emotions', {}))}")
        else:
            print("\nNo similar past responses found.")
    
    # Save to file if requested
    if args.output:
        export_to_json(response, args.output)
        print(f"\nOutput saved to {args.output}")
    
    # Print debug info if requested
    if args.debug:
        print("\nDebug Information:")
        debug_info = response.get("debug", {})
        # Filter out potentially large fields
        if "semantic_analysis" in debug_info:
            semantic_analysis = debug_info["semantic_analysis"]
            if "similar_responses" in semantic_analysis and len(semantic_analysis["similar_responses"]) > 0:
                print(f"Found {len(semantic_analysis['similar_responses'])} similar responses")
                semantic_analysis["similar_responses"] = f"[{len(semantic_analysis['similar_responses'])} responses]"
        print(json.dumps(debug_info, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0) 