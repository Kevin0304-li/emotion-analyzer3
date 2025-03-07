"""
Context Learning Module

This module enables the emotion calculator to learn from past 
interactions and improve response accuracy over time.
"""

import json
import os
import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict
import math

# Default path for storing interaction history
DEFAULT_HISTORY_PATH = "emotion_history.json"


class ContextLearner:
    """
    Learns from past emotional responses to improve future predictions.
    """
    
    def __init__(self, history_path: str = DEFAULT_HISTORY_PATH):
        """
        Initialize the context learner.
        
        Args:
            history_path: Path to store and load interaction history
        """
        self.history_path = history_path
        self.interaction_history = self._load_history()
        self.word_emotion_associations = defaultdict(lambda: defaultdict(float))
        self.phrase_emotion_associations = defaultdict(lambda: defaultdict(float))
        self.context_emotion_associations = defaultdict(lambda: defaultdict(float))
        
        # Rebuild associations from history
        self._rebuild_associations()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load interaction history from file."""
        if not os.path.exists(self.history_path):
            return []
            
        try:
            with open(self.history_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # Handle corrupted or unreadable file
            return []
    
    def _save_history(self) -> None:
        """Save interaction history to file."""
        with open(self.history_path, 'w') as f:
            json.dump(self.interaction_history, f, indent=2)
    
    def _rebuild_associations(self) -> None:
        """Rebuild associations from the interaction history."""
        self.word_emotion_associations.clear()
        self.phrase_emotion_associations.clear()
        self.context_emotion_associations.clear()
        
        for interaction in self.interaction_history:
            text = interaction.get("input_text", "").lower()
            emotions = interaction.get("emotions", {})
            context = interaction.get("context", "unknown")
            
            if not text or not emotions:
                continue
                
            # Update context-emotion associations
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else None
            if dominant_emotion:
                self.context_emotion_associations[context][dominant_emotion] += 1
            
            # Update word-emotion associations
            words = text.split()
            for word in words:
                for emotion, weight in emotions.items():
                    self.word_emotion_associations[word][emotion] += weight
            
            # Update phrase-emotion associations (bigrams)
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                for emotion, weight in emotions.items():
                    self.phrase_emotion_associations[bigram][emotion] += weight * 1.2  # Phrases weighted higher
    
    def add_interaction(self, 
                       text: str, 
                       emotions: Dict[str, float], 
                       context: str,
                       feedback: Optional[Dict[str, float]] = None) -> None:
        """
        Add a new interaction to the history.
        
        Args:
            text: Input text that generated the emotions
            emotions: Generated emotions and their intensities
            context: Relationship context (friend, enemy, neutral)
            feedback: Optional user feedback on emotions (for reinforcement)
        """
        timestamp = datetime.datetime.now().isoformat()
        
        interaction = {
            "timestamp": timestamp,
            "input_text": text,
            "emotions": emotions,
            "context": context
        }
        
        if feedback:
            interaction["feedback"] = feedback
        
        self.interaction_history.append(interaction)
        
        # Limit history size to prevent file growth
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]
            
        # Update associations
        self._rebuild_associations()
        
        # Save updated history
        self._save_history()
    
    def adjust_emotions(self, 
                       text: str, 
                       context: str, 
                       emotions: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust emotions based on learned patterns.
        
        Args:
            text: Input text 
            context: Relationship context
            emotions: Initial emotion predictions
            
        Returns:
            Adjusted emotions
        """
        if not self.interaction_history:
            return emotions  # No history to learn from
        
        text_lower = text.lower()
        words = text_lower.split()
        adjusted_emotions = emotions.copy()
        
        # Factor for how much learned patterns influence the result (0.0-1.0)
        learning_factor = min(0.3, len(self.interaction_history) / 100)  # Increases with more data
        
        # Adjust based on word-emotion associations
        word_adjustments = defaultdict(float)
        for word in words:
            if word in self.word_emotion_associations:
                word_emotions = self.word_emotion_associations[word]
                total_weight = sum(word_emotions.values())
                if total_weight > 0:
                    for emotion, weight in word_emotions.items():
                        normalized_weight = weight / total_weight
                        word_adjustments[emotion] += normalized_weight
        
        # Adjust based on phrase-emotion associations
        phrase_adjustments = defaultdict(float)
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in self.phrase_emotion_associations:
                phrase_emotions = self.phrase_emotion_associations[bigram]
                total_weight = sum(phrase_emotions.values())
                if total_weight > 0:
                    for emotion, weight in phrase_emotions.items():
                        normalized_weight = weight / total_weight
                        phrase_adjustments[emotion] += normalized_weight * 1.5  # Phrases weighted higher
        
        # Adjust based on context-emotion associations
        context_adjustments = defaultdict(float)
        if context in self.context_emotion_associations:
            context_emotions = self.context_emotion_associations[context]
            total_weight = sum(context_emotions.values())
            if total_weight > 0:
                for emotion, weight in context_emotions.items():
                    normalized_weight = weight / total_weight
                    context_adjustments[emotion] += normalized_weight
        
        # Apply adjustments
        for emotion in set(list(adjusted_emotions.keys()) + 
                          list(word_adjustments.keys()) + 
                          list(phrase_adjustments.keys()) + 
                          list(context_adjustments.keys())):
            
            # Base value (existing or 0)
            base_value = adjusted_emotions.get(emotion, 0.0)
            
            # Calculate adjustment
            adjustment = (
                word_adjustments.get(emotion, 0.0) * 0.4 +
                phrase_adjustments.get(emotion, 0.0) * 0.4 +
                context_adjustments.get(emotion, 0.0) * 0.2
            ) * learning_factor
            
            # Apply adjustment
            adjusted_emotions[emotion] = base_value * (1.0 - learning_factor) + adjustment
        
        # Normalize
        total = sum(adjusted_emotions.values())
        if total > 0:
            adjusted_emotions = {k: v / total for k, v in adjusted_emotions.items()}
        
        # Filter low values
        adjusted_emotions = {k: v for k, v in adjusted_emotions.items() if v >= 0.05}
        
        # Re-normalize
        total = sum(adjusted_emotions.values())
        if total > 0:
            adjusted_emotions = {k: v / total for k, v in adjusted_emotions.items()}
        
        return dict(sorted(adjusted_emotions.items(), key=lambda x: x[1], reverse=True))
    
    def get_similar_interactions(self, text: str, context: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar past interactions.
        
        Args:
            text: Input text to find similar interactions for
            context: Relationship context
            limit: Maximum number of similar interactions to return
            
        Returns:
            List of similar interactions
        """
        if not self.interaction_history:
            return []
            
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Calculate similarity scores
        similarity_scores = []
        for i, interaction in enumerate(self.interaction_history):
            past_text = interaction.get("input_text", "").lower()
            past_context = interaction.get("context", "unknown")
            
            # Skip if context is different
            if past_context != context:
                continue
                
            past_words = set(past_text.split())
            
            # Calculate word overlap
            common_words = words.intersection(past_words)
            if not common_words:
                continue
                
            # Jaccard similarity
            similarity = len(common_words) / len(words.union(past_words))
            
            # Boost for exact matches
            if text_lower == past_text:
                similarity = 1.0
                
            similarity_scores.append((i, similarity))
        
        # Sort by similarity and take top results
        similar_interactions = []
        for i, score in sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:limit]:
            similar_interactions.append(self.interaction_history[i])
            
        return similar_interactions


# Singleton instance
context_learner = ContextLearner()


def save_interaction(text: str, 
                    emotions: Dict[str, float], 
                    context: str,
                    feedback: Optional[Dict[str, float]] = None) -> None:
    """
    Save an interaction to the learning history.
    
    Args:
        text: Input text that generated the emotions
        emotions: Generated emotions and their intensities
        context: Relationship context (friend, enemy, neutral)
        feedback: Optional user feedback on emotions (for reinforcement)
    """
    context_learner.add_interaction(text, emotions, context, feedback)


def adjust_emotions_with_learning(
    text: str, 
    context: str, 
    emotions: Dict[str, float]
) -> Dict[str, float]:
    """
    Adjust emotions based on learned patterns.
    
    Args:
        text: Input text 
        context: Relationship context
        emotions: Initial emotion predictions
        
    Returns:
        Adjusted emotions
    """
    return context_learner.adjust_emotions(text, context, emotions)


def get_similar_emotional_responses(
    text: str, 
    context: str, 
    limit: int = 3
) -> List[Dict[str, Any]]:
    """
    Find similar past interactions.
    
    Args:
        text: Input text to find similar interactions for
        context: Relationship context
        limit: Maximum number of similar interactions to return
        
    Returns:
        List of similar interactions
    """
    return context_learner.get_similar_interactions(text, context, limit) 