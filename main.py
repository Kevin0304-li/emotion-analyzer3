"""
Emotion Calculator

Main entry point for the emotion calculator application.
"""

import argparse
import json
import sys
from typing import Dict, Any, Optional

from modules.integration import process_input, format_emotions, export_to_json


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
    
    return parser.parse_args()


def interactive_mode():
    """Run the emotion calculator in interactive mode."""
    print("=== Emotion Calculator Interactive Mode ===")
    print("Type 'exit' to quit.")
    
    # Keep track of conversation history for context
    history = {"entries": []}
    relationship = None
    
    # First, get the relationship context
    while relationship is None:
        rel_input = input("\nWhat is your relationship to the machine? (friend/enemy/neutral): ").strip().lower()
        if rel_input in ["friend", "enemy", "neutral"]:
            relationship = rel_input
        elif rel_input == "exit":
            return
        else:
            print("Please enter 'friend', 'enemy', or 'neutral'.")
    
    while True:
        # Get user input
        text = input("\nYou: ").strip()
        if text.lower() == "exit":
            break
        
        if not text:
            continue
            
        # Process the input
        response = process_input(
            text=text,
            relationship=relationship,
            history=history if history["entries"] else None
        )
        
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
        interactive_mode()
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
        debug=args.debug
    )
    
    # Display the results
    print("\nInput text:", args.text)
    print("Context:", response["context"])
    print("\nEmotional Response:")
    print(format_emotions(response["emotions"]))
    
    # Save to file if requested
    if args.output:
        export_to_json(response, args.output)
        print(f"\nOutput saved to {args.output}")
    
    # Print debug info if requested
    if args.debug:
        print("\nDebug Information:")
        print(json.dumps(response["debug"], indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0) 