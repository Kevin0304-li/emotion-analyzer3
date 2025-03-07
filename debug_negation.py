"""
Debug script for testing negation handling in semantic analysis.
"""

import spacy
from modules.semantic_analysis import analyze_text, analyze_sentiment_enhanced, SENTIMENT_LEXICON

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    nlp = spacy.load("en_core_web_sm")
    print("Using smaller spaCy model")

# Test phrases with negation
test_phrases = [
    "I don't hate this at all.",
    "I hate this.",
    "I don't like this.",
    "I like this.",
    "This is not bad.",
    "This is bad."
]

print("Analyzing test phrases for negation handling:")
print("-" * 50)

for phrase in test_phrases:
    print(f"\nPhrase: '{phrase}'")
    doc = nlp(phrase)
    
    # Look at each token and check if it's in our sentiment lexicons
    print("Tokens and their sentiment:")
    for token in doc:
        sentiment_found = False
        for sentiment_type, lexicon in SENTIMENT_LEXICON.items():
            if token.lemma_.lower() in lexicon:
                weight = lexicon[token.lemma_.lower()]
                print(f"  {token.text} ({token.lemma_.lower()}) - {sentiment_type}: {weight}")
                sentiment_found = True
        if not sentiment_found:
            print(f"  {token.text} - No sentiment")
    
    # Analyze sentiment with our enhanced method
    sentiment = analyze_sentiment_enhanced(doc)
    print("Sentiment scores:")
    for key, value in sentiment.items():
        print(f"  {key}: {value:.3f}")
    
    # Check if negation is working as expected
    if "hate" in phrase.lower() or "bad" in phrase.lower():
        if "not" in phrase.lower() or "don't" in phrase.lower():
            if sentiment["positive"] > sentiment["negative"]:
                print("✓ CORRECT: Negated negative sentiment is positive")
            else:
                print("✗ ERROR: Negated negative sentiment should be positive")
        else:
            if sentiment["negative"] > sentiment["positive"]:
                print("✓ CORRECT: Negative sentiment is negative")
            else:
                print("✗ ERROR: Negative sentiment should be negative")

# Print lexicon entries for reference
print("\n\nNegative words in our lexicon:")
for word, weight in sorted(SENTIMENT_LEXICON["negative"].items()):
    print(f"  {word}: {weight}")

print("\nPositive words in our lexicon:")
for word, weight in sorted(SENTIMENT_LEXICON["positive"].items())[:10]:  # Just show first 10
    print(f"  {word}: {weight}")

print("..." if len(SENTIMENT_LEXICON["positive"]) > 10 else "") 