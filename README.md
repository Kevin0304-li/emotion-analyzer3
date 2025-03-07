# Emotion Calculator

An AI-based emotion calculator that determines emotional responses based on text input and contextual relationships.

## Project Overview

This project analyzes input text and relationship context to generate a weighted distribution of emotions as a response. For example, when a user says "I'm going to kill you," the system will determine whether to respond with fear, excitement, or another emotion based on whether the speaker is considered a friend or an enemy.

## Project Structure

- `main.py`: Entry point for the application
- `modules/`: Core functionality modules
  - `semantic_analysis.py`: Processes user input, extracts semantic features, and analyzes sentiment
  - `context_judgment.py`: Determines the context (friend, enemy, neutral) from relationship info
  - `emotion_mapping.py`: Maps semantic features and context into a weighted emotion distribution
  - `integration.py`: Combines the outputs from the above modules
- `tests/`: Unit tests for the modules

## Requirements

See `requirements.txt` for a list of dependencies.

## Usage

```python
python main.py
```

## Example

Input: "I'm going to kill you" with context "friend"
Output: {"Happy": 0.5, "Excited": 0.3, "Fear": 0.2} 