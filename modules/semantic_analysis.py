"""
Semantic Analysis Module

This module processes user input text and extracts semantic features
along with sentiment scores using NLP techniques.
"""

import spacy
from typing import Dict, Any, Tuple, List, Set
import re

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


# Enhanced emotion lexicons with weights
SENTIMENT_LEXICON = {
    # Positive sentiment words with weights (0-1)
    "positive": {
        "good": 0.6, "great": 0.7, "excellent": 0.8, "happy": 0.7, "love": 0.9, 
        "like": 0.5, "nice": 0.6, "amazing": 0.8, "wonderful": 0.8, "joy": 0.7,
        "beautiful": 0.7, "fantastic": 0.8, "brilliant": 0.7, "awesome": 0.8,
        "enjoy": 0.6, "glad": 0.6, "pleased": 0.5, "delight": 0.7, "fabulous": 0.7,
        "superb": 0.8, "terrific": 0.7, "outstanding": 0.7, "perfect": 0.8,
        "positive": 0.6, "exciting": 0.6, "fun": 0.5, "lovely": 0.6, "pleasant": 0.5
    },
    
    # Negative sentiment words with weights (0-1)
    "negative": {
        "bad": 0.6, "terrible": 0.8, "awful": 0.8, "sad": 0.6, "hate": 0.9, 
        "dislike": 0.6, "horrible": 0.8, "disappointing": 0.6, "poor": 0.5,
        "ugly": 0.6, "stupid": 0.7, "annoying": 0.6, "unhappy": 0.7, "angry": 0.7,
        "depressed": 0.7, "frustrated": 0.6, "upset": 0.6, "furious": 0.8,
        "disgusting": 0.8, "pathetic": 0.7, "offensive": 0.7, "worthless": 0.7,
        "unpleasant": 0.6, "painful": 0.7, "miserable": 0.7, "rubbish": 0.6
    },
    
    # Hostile/threatening words with weights (0-1)
    "hostile": {
        "kill": 0.9, "hurt": 0.7, "destroy": 0.8, "hate": 0.7, "attack": 0.8, 
        "fight": 0.6, "die": 0.8, "murder": 0.9, "threat": 0.7, "harm": 0.7,
        "violent": 0.7, "slap": 0.6, "punch": 0.7, "kick": 0.6, "shoot": 0.8,
        "eliminate": 0.6, "torture": 0.8, "hit": 0.6, "crush": 0.7, "annihilate": 0.8,
        "stab": 0.8, "poison": 0.8, "assault": 0.8, "abuse": 0.7, "strangle": 0.8
    }
}

# Modifiers that intensify or diminish sentiment
INTENSIFIERS = {
    "very": 1.5, "extremely": 1.8, "really": 1.3, "incredibly": 1.6,
    "absolutely": 1.7, "definitely": 1.4, "totally": 1.5, "so": 1.3
}

DIMINISHERS = {
    "slightly": 0.5, "somewhat": 0.7, "a bit": 0.6, "kind of": 0.7,
    "sort of": 0.7, "barely": 0.3, "hardly": 0.3, "almost": 0.8
}

# Negations list - expanded to catch more variants
NEGATIONS = {
    "not", "no", "never", "none", "nothing", "neither", "nor", "nowhere",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
    "shouldn't", "mustn't", "without", "nobody", "n't", "nt"  # Adding standalone negation suffixes
}


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
        "is_question": any(token.tag_ == "?" for token in doc) or text.strip().endswith("?"),
        "sentence_count": len(list(doc.sents)),
        "imperative": is_imperative(doc),
        "exclamatory": "!" in text
    }
    
    # Enhanced sentiment analysis
    sentiment = analyze_sentiment_enhanced(doc)
    
    return {
        "features": features,
        "sentiment": sentiment
    }


def is_imperative(doc) -> bool:
    """
    Check if text appears to be a command or instruction.
    
    Args:
        doc: spaCy document
        
    Returns:
        Boolean indicating if text is likely imperative
    """
    # Simple heuristic: a sentence starting with a verb is likely imperative
    for sent in doc.sents:
        first_token = sent[0]
        if first_token.pos_ == "VERB":
            return True
            
    return False


def analyze_sentiment_enhanced(doc) -> Dict[str, float]:
    """
    Perform enhanced sentiment analysis using lexicons with weights.
    
    Args:
        doc: spaCy document
        
    Returns:
        Dictionary with detailed sentiment scores
    """
    sentiment_scores = {
        "positive": 0.0,
        "negative": 0.0,
        "neutral": 0.0,
        "hostility": 0.0,
        "intensity": 0.0  # Overall emotional intensity
    }
    
    if len(doc) == 0:
        sentiment_scores["neutral"] = 1.0
        return sentiment_scores
    
    # Track negation context
    negation_active = False
    negation_window = 0  # Track window of words affected by negation
    negation_window_size = 4  # Number of words after negation to apply negation effect
    skip_indices = set()  # Track indices of tokens to skip
    
    # Check for contractions with negations first
    text_lower = doc.text.lower()
    # If the full text contains common negated terms, set a flag for special handling
    has_negation_phrases = any(neg in text_lower for neg in [
        "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", 
        "weren't", "haven't", "hasn't", "hadn't", "can't", "couldn't", 
        "wouldn't", "shouldn't", "won't", "not"
    ])
    
    # Process text for sentiment
    for i, token in enumerate(doc):
        if i in skip_indices:
            continue
        
        # Enhanced negation detection
        if (token.lower_ in NEGATIONS or 
            (token.text.lower().endswith("n't") or token.text.lower().endswith("nt")) or
            (i > 0 and doc[i-1].text.lower() + token.text.lower() in NEGATIONS)):
            negation_active = True
            negation_window = 0
            continue
            
        # Reset negation after a certain number of tokens or at punctuation
        if negation_active:
            negation_window += 1
            if negation_window >= negation_window_size or token.is_punct:
                negation_active = False
            
        # Get the lemmatized lowercased word
        word = token.lemma_.lower()
        
        # Special handling for common negated sentiment words
        if has_negation_phrases and not negation_active:
            # Check if this token is part of a negated phrase
            for j in range(max(0, i-2), i):  # Look at previous 2 tokens
                if j < len(doc) and (doc[j].lower_ in NEGATIONS or doc[j].text.lower().endswith("n't")):
                    # This word might be affected by a negation
                    negation_active = True
                    break
                    
        # Look for bigrams/trigrams (multi-word expressions)
        if i < len(doc) - 1:
            bigram = f"{word} {doc[i+1].lemma_.lower()}"
            
            # Check for intensifiers and diminishers in bigrams
            if bigram in DIMINISHERS:
                skip_indices.add(i+1)
                if i+2 < len(doc):
                    next_word = doc[i+2].lemma_.lower()
                    process_sentiment_word(next_word, sentiment_scores, 
                                          negation_active, diminisher=DIMINISHERS[bigram])
                    skip_indices.add(i+2)
                continue
                
            if bigram in INTENSIFIERS:
                skip_indices.add(i+1)
                if i+2 < len(doc):
                    next_word = doc[i+2].lemma_.lower()
                    process_sentiment_word(next_word, sentiment_scores, 
                                          negation_active, intensifier=INTENSIFIERS[bigram])
                    skip_indices.add(i+2)
                continue
                
        # Check if the current word is an intensifier or diminisher
        if word in INTENSIFIERS and i+1 < len(doc):
            intensifier = INTENSIFIERS[word]
            next_word = doc[i+1].lemma_.lower()
            process_sentiment_word(next_word, sentiment_scores, negation_active, intensifier=intensifier)
            skip_indices.add(i+1)
            continue
            
        if word in DIMINISHERS and i+1 < len(doc):
            diminisher = DIMINISHERS[word]
            next_word = doc[i+1].lemma_.lower()
            process_sentiment_word(next_word, sentiment_scores, negation_active, diminisher=diminisher)
            skip_indices.add(i+1)
            continue
                
        # Process individual words
        process_sentiment_word(word, sentiment_scores, negation_active)
    
    # Special case handling for phrases like "don't hate" that weren't caught
    if has_negation_phrases:
        for sentiment_type, lexicon in SENTIMENT_LEXICON.items():
            for word in lexicon:
                negated_patterns = [f"don't {word}", f"doesn't {word}", f"didn't {word}", 
                                   f"isn't {word}", f"not {word}", f"never {word}"]
                for pattern in negated_patterns:
                    if pattern in text_lower:
                        # Found a negated sentiment word
                        weight = lexicon[word]
                        if sentiment_type == "negative":
                            # Negate the negative sentiment
                            sentiment_scores["positive"] += weight
                            sentiment_scores["negative"] = max(0, sentiment_scores["negative"] - weight * 0.7)
                        elif sentiment_type == "positive":
                            # Negate the positive sentiment
                            sentiment_scores["negative"] += weight * 0.7
                            sentiment_scores["positive"] = max(0, sentiment_scores["positive"] - weight * 0.5)
                        elif sentiment_type == "hostile":
                            # Negate hostility
                            sentiment_scores["hostility"] = max(0, sentiment_scores["hostility"] - weight * 0.6)
    
    # Calculate neutral score
    total_non_neutral = sentiment_scores["positive"] + sentiment_scores["negative"]
    sentiment_scores["neutral"] = max(0.0, 1.0 - total_non_neutral)
    
    # Calculate total emotional intensity
    sentiment_scores["intensity"] = total_non_neutral + sentiment_scores["hostility"]
    sentiment_scores["intensity"] = min(1.0, sentiment_scores["intensity"])
    
    # Normalize scores to 0-1 range
    total_words = len([t for t in doc if not t.is_punct and not t.is_space])
    if total_words > 0:
        for key in sentiment_scores:
            sentiment_scores[key] = min(1.0, sentiment_scores[key])
    
    return sentiment_scores


def process_sentiment_word(word: str, scores: Dict[str, float], negated: bool, 
                          intensifier: float = None, diminisher: float = None) -> None:
    """
    Process a single word for sentiment and update scores.
    
    Args:
        word: The word to process
        scores: Dictionary of sentiment scores to update
        negated: Whether the word is in a negation context
        intensifier: Optional intensity multiplier
        diminisher: Optional intensity reducer
    """
    # Check if word is in any of our sentiment lexicons
    modifier = 1.0
    if intensifier:
        modifier = intensifier
    elif diminisher:
        modifier = diminisher
    
    # Apply the sentiment based on lexicons
    for sentiment_type, lexicon in SENTIMENT_LEXICON.items():
        if word in lexicon:
            weight = lexicon[word] * modifier
            
            # Map 'hostile' sentiment type to 'hostility' score key
            score_key = "hostility" if sentiment_type == "hostile" else sentiment_type
            
            # Negation flips sentiment between positive and negative
            if negated:
                if sentiment_type == "positive":
                    scores["negative"] += weight
                elif sentiment_type == "negative":
                    # Stronger conversion of negated negative to positive
                    scores["positive"] += weight  # Full weight for negated negative
                    scores["negative"] -= weight * 0.5  # Also reduce the negative score
                else:
                    # Hostility is reduced by negation
                    scores[score_key] += weight * 0.3  # More reduction in hostile sentiment when negated
            else:
                scores[score_key] += weight
    
    # Ensure no negative values in scores
    for key in scores:
        scores[key] = max(0.0, scores[key])


if __name__ == "__main__":
    # Example usage
    test_text = "I'm going to kill you"
    results = analyze_text(test_text)
    print(f"Analysis of '{test_text}':")
    print(f"Features: {results['features']}")
    print(f"Sentiment: {results['sentiment']}")
    
    # Test negation and intensifiers
    test_text2 = "I really don't like you at all"
    results2 = analyze_text(test_text2)
    print(f"\nAnalysis of '{test_text2}':")
    print(f"Sentiment: {results2['sentiment']}") 