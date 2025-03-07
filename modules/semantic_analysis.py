"""
Semantic Analysis Module

This module processes user input text and extracts semantic features
along with sentiment scores using NLP techniques.
"""

import spacy
from typing import Dict, Any, Tuple

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    # If the model isn't installed, display a message about how to install it
    print("Spacy model 'en_core_web_md' not found. Please install it using:")
    print("python -m spacy download en_core_web_md")
    # Fall back to the small model which might be already installed
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Using 'en_core_web_sm' as fallback")
    except OSError:
        print("Spacy model 'en_core_web_sm' not found either. Please install it using:")
        print("python -m spacy download en_core_web_sm")
        nlp = None


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze the input text to extract semantic features and sentiment.
    
    Args:
        text: Input text string to analyze
        
    Returns:
        Dictionary containing semantic features and sentiment analysis
    """
    if nlp is None:
        return {"error": "Spacy model not loaded. Please install a model first."}
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Extract key features
    features = {
        "entities": [{"text": ent.text, "label": ent.label_} for ent in doc.ents],
        "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
        "key_words": [token.text for token in doc if token.pos_ in ("VERB", "NOUN", "ADJ") and not token.is_stop],
        "word_count": len([token for token in doc if not token.is_punct and not token.is_space]),
        "is_question": any(token.tag_ == "?" for token in doc),
    }
    
    # Simple rule-based sentiment analysis
    sentiment = analyze_sentiment(doc)
    
    return {
        "features": features,
        "sentiment": sentiment
    }


def analyze_sentiment(doc) -> Dict[str, float]:
    """
    Perform simple rule-based sentiment analysis.
    
    Args:
        doc: spaCy document
        
    Returns:
        Dictionary with sentiment scores
    """
    # Count positive and negative words based on simple lexicon
    # This is a very basic approach - in a real system you'd use a more sophisticated sentiment model
    positive_words = {"good", "great", "excellent", "happy", "love", "like", "nice", "amazing", "wonderful"}
    negative_words = {"bad", "terrible", "awful", "sad", "hate", "dislike", "horrible", "disappointing"}
    hostile_words = {"kill", "hurt", "destroy", "hate", "attack", "fight", "die"}
    
    # Count occurrences
    pos_count = sum(1 for token in doc if token.lemma_.lower() in positive_words)
    neg_count = sum(1 for token in doc if token.lemma_.lower() in negative_words)
    hostile_count = sum(1 for token in doc if token.lemma_.lower() in hostile_words)
    
    # Calculate base sentiment
    total_words = len(doc)
    if total_words == 0:
        return {"positive": 0, "negative": 0, "neutral": 1.0, "hostility": 0}
    
    positive_score = pos_count / total_words
    negative_score = neg_count / total_words
    hostility_score = hostile_count / total_words
    neutral_score = 1.0 - (positive_score + negative_score)
    
    return {
        "positive": positive_score,
        "negative": negative_score,
        "neutral": neutral_score,
        "hostility": hostility_score
    }


if __name__ == "__main__":
    # Example usage
    test_text = "I'm going to kill you"
    results = analyze_text(test_text)
    print(f"Analysis of '{test_text}':")
    print(f"Features: {results['features']}")
    print(f"Sentiment: {results['sentiment']}") 