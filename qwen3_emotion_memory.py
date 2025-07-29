import torch
import os
import socket
import time
import hashlib
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import requests
import datetime
import spacy
import uuid
import re
import asyncio
import concurrent.futures
from dataclasses import dataclass
import json
import random

#https://claude.ai/chat/a81a01ef-e9ee-431e-944d-df7ed9f19f77

"""
Contextual LLM Chat Flow
========================

An intelligent chat system that uses parallel processing to analyze input text,
extract emotional context and memories, then dynamically shapes LLM responses.

Architecture:
1. Input Processing: Extract nouns from user input using spaCy
2. Parallel Analysis: Simultaneously analyze emotions and search memories
3. Integration: Combine insights to create dynamic system prompts
4. Response Generation: Use Qwen with contextually-aware prompting

Dependencies:
- QwenChat class (for LLM interaction)
- MxBaiEmbedder class (for embeddings and emotion analysis)
- SpacyNounExtractor class (for noun extraction)
"""



@dataclass
class AnalysisResult:
    """Container for parallel analysis results"""
    nouns: List[str]
    emotions: List[Tuple[str, float, Dict]]
    memories: List[Tuple[str, float, str]]
    processing_time: float
    context_strength: float

# Plutchik's primary emotions with their characteristics
EMOTION_DATABASE = {
    # Joy family
    'joy': {
        'mood': 'Sense of energy and possibility',
        'thoughts': 'Life is going well',
        'responses': 'Sparks creativity, connection, gives energy'
    },
    'ecstasy': {
        'mood': 'Overwhelming euphoria and elation',
        'thoughts': 'Everything is perfect and amazing',
        'responses': 'Boundless enthusiasm, may act impulsively from excitement'
    },
    'serenity': {
        'mood': 'Calm contentment and peace',
        'thoughts': 'Things are pleasant and stable',
        'responses': 'Gentle actions, seeks to maintain harmony'
    },
    
    # Sadness family
    'sadness': {
        'mood': 'Heavy, low energy, withdrawn',
        'thoughts': 'Things aren\'t going well, feeling loss',
        'responses': 'Seeks comfort, may isolate, moves slowly'
    },
    'grief': {
        'mood': 'Profound sorrow and despair',
        'thoughts': 'Something important is gone forever',
        'responses': 'May be inconsolable, needs support, difficulty functioning'
    },
    'pensiveness': {
        'mood': 'Quiet melancholy and reflection',
        'thoughts': 'Contemplating what could have been',
        'responses': 'Introspective, seeks solitude, gentle sadness'
    },
    
    # Trust family
    'trust': {
        'mood': 'Open and accepting',
        'thoughts': 'Others are reliable and good',
        'responses': 'Cooperative, shares freely, seeks connection'
    },
    'admiration': {
        'mood': 'Deep respect and reverence',
        'thoughts': 'This person/thing is truly worthy',
        'responses': 'Wants to learn, emulate, or serve'
    },
    'acceptance': {
        'mood': 'Calm acknowledgment',
        'thoughts': 'This is how things are',
        'responses': 'Goes with the flow, doesn\'t resist'
    },
    
    # Disgust family
    'disgust': {
        'mood': 'Repulsed and rejecting',
        'thoughts': 'This is wrong, contaminated, or inferior',
        'responses': 'Avoids, criticizes, seeks to remove or cleanse'
    },
    'loathing': {
        'mood': 'Intense revulsion and hatred',
        'thoughts': 'This is absolutely abhorrent',
        'responses': 'Strong rejection, may become aggressive to eliminate'
    },
    'boredom': {
        'mood': 'Mild disinterest and restlessness',
        'thoughts': 'This isn\'t worth my attention',
        'responses': 'Seeks stimulation elsewhere, disengages'
    },
    
    # Fear family
    'fear': {
        'mood': 'Anxious alertness and tension',
        'thoughts': 'Something bad might happen',
        'responses': 'Cautious, seeks safety, may freeze or flee'
    },
    'terror': {
        'mood': 'Paralyzing dread',
        'thoughts': 'Immediate danger, might not survive',
        'responses': 'Fight, flight, or freeze response, acts on instinct'
    },
    'apprehension': {
        'mood': 'Mild worry and uncertainty',
        'thoughts': 'Something doesn\'t feel quite right',
        'responses': 'More cautious than usual, seeks reassurance'
    },
    
    # Anger family
    'anger': {
        'mood': 'Heated and energized',
        'thoughts': 'This is unfair, I\'ve been wronged',
        'responses': 'Confrontational, seeks to correct or punish'
    },
    'rage': {
        'mood': 'Burning fury and aggression',
        'thoughts': 'Must destroy the source of this injustice',
        'responses': 'Potentially violent, loses rational control'
    },
    'annoyance': {
        'mood': 'Mildly irritated and impatient',
        'thoughts': 'This is inconvenient or bothersome',
        'responses': 'Short responses, may express frustration verbally'
    },
    
    # Surprise family
    'surprise': {
        'mood': 'Startled and alert',
        'thoughts': 'That was unexpected',
        'responses': 'Heightened attention, pauses to process'
    },
    'amazement': {
        'mood': 'Awed and wonder-struck',
        'thoughts': 'This is incredible and beyond belief',
        'responses': 'Stares, asks questions, wants to understand'
    },
    'distraction': {
        'mood': 'Mildly surprised and unfocused',
        'thoughts': 'Wait, what was that?',
        'responses': 'Attention shifts, momentarily loses focus'
    },
    
    # Anticipation family
    'anticipation': {
        'mood': 'Eager and forward-looking',
        'thoughts': 'Something good is coming',
        'responses': 'Prepares, plans, may act impatiently'
    },
    'vigilance': {
        'mood': 'Intense focus and readiness',
        'thoughts': 'Must be ready for what\'s coming',
        'responses': 'Hyper-alert, prepared for action'
    },
    'interest': {
        'mood': 'Curious and engaged',
        'thoughts': 'I want to know more about this',
        'responses': 'Asks questions, explores, pays attention'
    }
}

# Complex emotions formed by combining primary emotions
COMPLEX_EMOTIONS = {
    'love': {
        'components': ['joy', 'trust'],
        'mood': 'Warm, connected, and devoted',
        'thoughts': 'This person/thing is wonderful and safe',
        'responses': 'Protective, nurturing, wants to be close'
    },
    'submission': {
        'components': ['trust', 'fear'],
        'mood': 'Deferential and compliant',
        'thoughts': 'I should follow their lead',
        'responses': 'Obeys, seeks approval, avoids conflict'
    },
    'awe': {
        'components': ['fear', 'surprise'],
        'mood': 'Humbled and overwhelmed',
        'thoughts': 'This is beyond my understanding',
        'responses': 'Reverent behavior, may feel small or insignificant'
    },
    'disapproval': {
        'components': ['surprise', 'sadness'],
        'mood': 'Disappointed and let down',
        'thoughts': 'This isn\'t what I expected or hoped for',
        'responses': 'Expresses dissatisfaction, may withdraw support'
    },
    'remorse': {
        'components': ['sadness', 'disgust'],
        'mood': 'Regretful and self-reproaching',
        'thoughts': 'I did something wrong and feel bad about it',
        'responses': 'Apologizes, seeks to make amends, self-punishing'
    },
    'contempt': {
        'components': ['disgust', 'anger'],
        'mood': 'Superior and disdainful',
        'thoughts': 'This is beneath me and doesn\'t deserve respect',
        'responses': 'Dismissive, condescending, may ridicule'
    },
    'aggressiveness': {
        'components': ['anger', 'anticipation'],
        'mood': 'Hostile and ready for conflict',
        'thoughts': 'I need to attack before they do',
        'responses': 'Threatening behavior, seeks confrontation'
    },
    'optimism': {
        'components': ['anticipation', 'joy'],
        'mood': 'Hopeful and positive about the future',
        'thoughts': 'Good things are coming',
        'responses': 'Plans enthusiastically, encourages others'
    }
}


class SpacyNounExtractor:
    """
    A class for extracting nouns and analyzing text using spaCy.
    
    Installation:
        pip install spacy
        python -m spacy download en_core_web_sm
    
    Usage:
        extractor = SpacyNounExtractor()
        nouns = extractor.find_nouns("The cat sat on the mat.")
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the noun extractor with a spaCy model.
        
        Args:
            model_name (str): Name of the spaCy model to load
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"spaCy model '{model_name}' not found. "
                f"Install with: python -m spacy download {model_name}"
            )
    
    def find_nouns(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find all nouns (regular and proper) in the text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str, str]]: List of (word, pos_tag, detailed_tag)
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_, token.tag_) 
                for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    def find_regular_nouns(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find only regular nouns (not proper nouns).
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str, str]]: List of (word, pos_tag, detailed_tag)
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_, token.tag_) 
                for token in doc if token.pos_ == "NOUN"]
    
    def find_proper_nouns(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find only proper nouns.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str, str]]: List of (word, pos_tag, detailed_tag)
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_, token.tag_) 
                for token in doc if token.pos_ == "PROPN"]
    
    def find_named_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Find named entities with their categories.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Dict[str, str]]: List of entity dictionaries
        """
        doc = self.nlp(text)
        return [
            {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_) or 'Unknown'
            }
            for ent in doc.ents
        ]
    
    def find_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases (multi-word noun expressions).
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[str]: List of noun phrases
        """
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
    
    def get_noun_lemmas(self, text: str) -> List[Tuple[str, str]]:
        """
        Get nouns with their lemmatized (base) forms.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str]]: List of (original_word, lemma)
        """
        doc = self.nlp(text)
        return [(token.text, token.lemma_) 
                for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    def analyze_text(self, text: str) -> Dict[str, Union[List, Dict]]:
        """
        Perform comprehensive noun analysis of the text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict: Comprehensive analysis results
        """
        doc = self.nlp(text)
        
        analysis = {
            'text': text,
            'regular_nouns': [],
            'proper_nouns': [],
            'named_entities': [],
            'noun_phrases': [],
            'noun_lemmas': {},
            'statistics': {}
        }
        
        # Collect different types of nouns
        for token in doc:
            if token.pos_ == "NOUN":
                analysis['regular_nouns'].append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tag': token.tag_
                })
            elif token.pos_ == "PROPN":
                analysis['proper_nouns'].append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tag': token.tag_
                })
        
        # Named entities
        analysis['named_entities'] = [
            {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_) or 'Unknown'
            }
            for ent in doc.ents
        ]
        
        # Noun phrases
        analysis['noun_phrases'] = [chunk.text for chunk in doc.noun_chunks]
        
        # Noun lemmas mapping
        analysis['noun_lemmas'] = {
            token.text: token.lemma_ 
            for token in doc if token.pos_ in ["NOUN", "PROPN"]
        }
        
        # Statistics
        analysis['statistics'] = {
            'total_nouns': len(analysis['regular_nouns']) + len(analysis['proper_nouns']),
            'regular_noun_count': len(analysis['regular_nouns']),
            'proper_noun_count': len(analysis['proper_nouns']),
            'entity_count': len(analysis['named_entities']),
            'noun_phrase_count': len(analysis['noun_phrases'])
        }
        
        return analysis
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        Efficiently analyze multiple texts at once.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Dict]: List of analysis results for each text
        """
        docs = list(self.nlp.pipe(texts))
        results = []
        
        for text, doc in zip(texts, docs):
            nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
            entities = [
                {'text': ent.text, 'label': ent.label_}
                for ent in doc.ents
            ]
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            results.append({
                'text': text,
                'nouns': nouns,
                'entities': entities,
                'noun_phrases': noun_phrases
            })
        
        return results
    
    def get_simple_nouns(self, text: str) -> List[str]:
        """
        Get just the noun words as a simple list.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[str]: Simple list of noun words
        """
        doc = self.nlp(text)
        return [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    def filter_nouns_by_type(self, text: str, noun_types: List[str]) -> List[str]:
        """
        Filter nouns by specific POS types.
        
        Args:
            text (str): Input text to analyze
            noun_types (List[str]): List of POS types to include (e.g., ['NOUN'], ['PROPN'])
            
        Returns:
            List[str]: Filtered list of nouns
        """
        doc = self.nlp(text)
        return [token.text for token in doc if token.pos_ in noun_types]
    
    @staticmethod
    def explain_pos_tags() -> Dict[str, str]:
        """
        Get explanations for noun POS tags.
        
        Returns:
            Dict[str, str]: Dictionary of POS tags and their descriptions
        """
        return {
            'NOUN': 'Common noun (dog, car, book, freedom)',
            'PROPN': 'Proper noun (John, London, Apple)',
            'NN': 'Noun, singular',
            'NNS': 'Noun, plural',
            'NNP': 'Proper noun, singular',
            'NNPS': 'Proper noun, plural'
        }



class MxBaiEmbedder:
    def __init__(self, pickle_file: str = "embeddings_store.pkl"):
        self.tokenizer = None
        self.model = None
        self.embeddings_store = {}  # Dict to store embeddings with UUID keys
        self.metadata_store = {}    # Store original text and other metadata
        self.pickle_file = pickle_file
        self.emotions_initialized = False
        
        # Load existing embeddings if pickle file exists
        self._load_from_pickle()
        
    def check_internet_connection(self, timeout=5):
        """Check if internet connection is available"""
        try:
            requests.get("https://www.google.com", timeout=timeout)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False

    def _load_from_pickle(self):
        """Load embeddings and metadata from pickle file"""
        if os.path.exists(self.pickle_file):
            try:
                with open(self.pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings_store = data.get('embeddings_store', {})
                    self.metadata_store = data.get('metadata_store', {})
                    self.emotions_initialized = data.get('emotions_initialized', False)
                print(f"Loaded {len(self.embeddings_store)} embeddings from {self.pickle_file}")
                if self.emotions_initialized:
                    print("Emotion embeddings already initialized")
            except Exception as e:
                print(f"Error loading from pickle file: {str(e)}")
                print("Starting with empty embeddings store.")
                self.embeddings_store = {}
                self.metadata_store = {}
                self.emotions_initialized = False
        else:
            print(f"No existing pickle file found at {self.pickle_file}. Starting with empty store.")

    def _save_to_pickle(self):
        """Save embeddings and metadata to pickle file"""
        try:
            data = {
                'embeddings_store': self.embeddings_store,
                'metadata_store': self.metadata_store,
                'emotions_initialized': self.emotions_initialized
            }
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {len(self.embeddings_store)} embeddings to {self.pickle_file}")
        except Exception as e:
            print(f"Error saving to pickle file: {str(e)}")

    def load_model(self):
        """Load mixedbread-ai/mxbai-embed-large-v1 model with offline fallback"""
        model_name = "mixedbread-ai/mxbai-embed-large-v1"
        
        # Check internet connection
        has_internet = self.check_internet_connection()
        
        if not has_internet:
            print("No internet connection detected. Setting offline mode...")
            # Set offline environment variables
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
        else:
            print("Internet connection available. Loading model online...")
            # Ensure offline flags are not set
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            os.environ.pop('HF_DATASETS_OFFLINE', None)
        
        try:
            # Load tokenizer and model
            print(f"Loading tokenizer for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"Loading model {model_name}...")
            self.model = AutoModel.from_pretrained(model_name)
            
            # Set model to evaluation mode
            self.model.eval()
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            if has_internet:
                print("Failed to load online. Trying offline mode...")
                # Set offline flags and retry
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(model_name)
                    self.model.eval()
                    print("Model loaded successfully from cache!")
                    return True
                except Exception as offline_e:
                    print(f"Failed to load from cache: {str(offline_e)}")
                    return False
            else:
                print("No internet connection and cache loading failed.")
                return False

    def embed_text_string(self, text: str) -> np.ndarray:
        """
        Create an embedding for a given text string
        
        Args:
            text (str): The input text to embed
            
        Returns:
            np.ndarray: The embedding vector
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                               padding=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use mean pooling of the last hidden state
            # This is a common approach for sentence embeddings
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Apply attention mask and mean pooling
            masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
            summed_embeddings = torch.sum(masked_embeddings, dim=1)
            summed_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = summed_embeddings / summed_mask.unsqueeze(-1)
            
            # Convert to numpy array
            embedding_vector = mean_pooled.squeeze().numpy()
            
            # Normalize the embedding (optional but often helpful for similarity search)
            embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
            
            return embedding_vector

    def generate_content_id(self, text: str) -> str:
        """Generate deterministic ID based on content
        THIS WILL AUTO overwrite duplicates. So it's impossible to save enter the same thing twice. con being counting and same text different meta data"""
        return hashlib.md5(text.encode()).hexdigest()

    def initialize_emotion_embeddings(self, force_reinitialize: bool = False):
        """
        Initialize embeddings for all emotions in the database
        
        Args:
            force_reinitialize (bool): If True, will re-embed emotions even if already done
        """
        if self.emotions_initialized and not force_reinitialize:
            print("Emotion embeddings already initialized. Use force_reinitialize=True to recreate them.")
            return
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Initializing emotion embeddings...")
        
        # Combine all emotions
        all_emotions = {**EMOTION_DATABASE, **COMPLEX_EMOTIONS}
        
        emotion_texts = []
        emotion_ids = []
        
        for emotion_name, emotion_data in all_emotions.items():
            # Create a comprehensive text representation of the emotion
            if 'components' in emotion_data:
                # Complex emotion
                emotion_text = f"emotion: {emotion_name}. components: {', '.join(emotion_data['components'])}. mood: {emotion_data['mood']}. thoughts: {emotion_data['thoughts']}. responses: {emotion_data['responses']}"
            else:
                # Primary emotion
                emotion_text = f"emotion: {emotion_name}. mood: {emotion_data['mood']}. thoughts: {emotion_data['thoughts']}. responses: {emotion_data['responses']}"
            
            emotion_texts.append(emotion_text)
            emotion_ids.append(f"emotion_{emotion_name}")
        
        # Batch embed all emotions
        print(f"Embedding {len(emotion_texts)} emotions...")
        for i, (text, emotion_id) in enumerate(zip(emotion_texts, emotion_ids)):
            embedding = self.embed_text_string(text)
            
            # Store with special metadata to mark as emotion
            self.embeddings_store[emotion_id] = embedding
            self.metadata_store[emotion_id] = {
                'text': text,
                'embedding_shape': embedding.shape,
                'created_at': str(uuid.uuid1().time),
                'type': 'emotion',
                'emotion_name': emotion_id.replace('emotion_', ''),
                'emotion_data': all_emotions[emotion_id.replace('emotion_', '')]
            }
            
            if (i + 1) % 5 == 0:
                print(f"Embedded {i + 1}/{len(emotion_texts)} emotions...")
        
        self.emotions_initialized = True
        self._save_to_pickle()
        print(f"Successfully initialized {len(emotion_texts)} emotion embeddings!")

    def find_most_similar_emotion(self, text: str) -> Tuple[str, float, Dict]:
        """
        Find the most similar emotion to the given text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple[str, float, Dict]: (emotion_name, similarity_score, emotion_attributes)
        """
        if not self.emotions_initialized:
            raise ValueError("Emotion embeddings not initialized. Call initialize_emotion_embeddings() first.")
        
        # Get embedding for the input text
        query_embedding = self.embed_text_string(text)
        
        # Filter only emotion embeddings
        emotion_ids = [eid for eid in self.embeddings_store.keys() if eid.startswith('emotion_')]
        
        if not emotion_ids:
            raise ValueError("No emotion embeddings found in store.")
        
        # Get emotion embeddings
        emotion_embeddings = np.array([self.embeddings_store[eid] for eid in emotion_ids])
        
        # Ensure query embedding is 2D for sklearn
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, emotion_embeddings)[0]
        
        # Find the most similar emotion
        best_idx = np.argmax(similarities)
        best_emotion_id = emotion_ids[best_idx]
        best_similarity = float(similarities[best_idx])
        
        # Get emotion data
        emotion_name = best_emotion_id.replace('emotion_', '')
        emotion_data = self.metadata_store[best_emotion_id]['emotion_data']
        
        return emotion_name, best_similarity, emotion_data

    def analyze_text_emotion(self, text: str, top_n: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        Analyze text and return top N most similar emotions with their attributes
        
        Args:
            text (str): Input text to analyze
            top_n (int): Number of top emotions to return
            
        Returns:
            List[Tuple[str, float, Dict]]: List of (emotion_name, similarity_score, emotion_attributes)
        """
        if not self.emotions_initialized:
            raise ValueError("Emotion embeddings not initialized. Call initialize_emotion_embeddings() first.")
        
        # Get embedding for the input text
        query_embedding = self.embed_text_string(text)
        
        # Filter only emotion embeddings
        emotion_ids = [eid for eid in self.embeddings_store.keys() if eid.startswith('emotion_')]
        
        if not emotion_ids:
            raise ValueError("No emotion embeddings found in store.")
        
        # Get emotion embeddings
        emotion_embeddings = np.array([self.embeddings_store[eid] for eid in emotion_ids])
        
        # Ensure query embedding is 2D for sklearn
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, emotion_embeddings)[0]
        
        # Get top N emotions
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            emotion_id = emotion_ids[idx]
            similarity_score = float(similarities[idx])
            emotion_name = emotion_id.replace('emotion_', '')
            emotion_data = self.metadata_store[emotion_id]['emotion_data']
            results.append((emotion_name, similarity_score, emotion_data))
        
        return results

    def save_embedding(self, text: str, embedding: Optional[np.ndarray] = None, 
                      custom_id: Optional[str] = None, auto_save: bool = True,
                      check_duplicates: bool = True) -> str:
        """
        Save an embedding to the in-memory store and optionally persist to disk
        
        Args:
            text (str): Original text
            embedding (np.ndarray, optional): Precomputed embedding. If None, will compute it.
            custom_id (str, optional): Custom ID. If None, will generate UUID.
            auto_save (bool): Whether to automatically save to pickle file
            check_duplicates (bool): make sure we haven't embedded the same exact text before
            
        Returns:
            str: The ID used to store the embedding
        """            
        # Generate ID if not provided
        if custom_id is None:
            #embedding_id = str(uuid.uuid4())
            embedding_id = self.generate_content_id(text)
        else:
            embedding_id = custom_id
            
        # Compute embedding if not provided
        if embedding is None:
            embedding = self.embed_text_string(text)
            
        # Store embedding and metadata
        self.embeddings_store[embedding_id] = embedding
        self.metadata_store[embedding_id] = {
            'text': text,
            'embedding_shape': embedding.shape,
            'created_at': str(uuid.uuid1().time),
            'type': 'general'
        }
        
        # Auto-save to pickle file if requested
        if auto_save:
            self._save_to_pickle()
        
        return embedding_id

    def search_embeddings(self, query_embedding: np.ndarray, n: int = 5, 
                         exclude_emotions: bool = False) -> List[Tuple[str, float, str]]:
        """
        Search for the most similar embeddings
        
        Args:
            query_embedding (np.ndarray): The query embedding vector
            n (int): Number of most similar embeddings to return
            exclude_emotions (bool): Whether to exclude emotion embeddings from search
            
        Returns:
            List[Tuple[str, float, str]]: List of (embedding_id, similarity_score, original_text)
        """
        if exclude_emotions:
            # Filter out emotion embeddings
            available_ids = [eid for eid in self.embeddings_store.keys() 
                           if not eid.startswith('emotion_')]
        else:
            available_ids = list(self.embeddings_store.keys())
        
        if len(available_ids) == 0:
            return []
            
        # Get stored embeddings
        stored_embeddings = np.array([self.embeddings_store[id_] for id_ in available_ids])
        
        # Ensure query embedding is 2D for sklearn
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
        
        # Get top n similar embeddings
        top_indices = np.argsort(similarities)[::-1][:n]
        
        results = []
        for idx in top_indices:
            embedding_id = available_ids[idx]
            similarity_score = similarities[idx]
            original_text = self.metadata_store[embedding_id]['text']
            results.append((embedding_id, float(similarity_score), original_text))
            
        return results

    def search_by_text(self, query_text: str, n: int = 5, 
                      exclude_emotions: bool = False) -> List[Tuple[str, float, str]]:
        """
        Convenience method to search by text (will compute embedding first)
        
        Args:
            query_text (str): The query text
            n (int): Number of most similar embeddings to return
            exclude_emotions (bool): Whether to exclude emotion embeddings from search
            
        Returns:
            List[Tuple[str, float, str]]: List of (embedding_id, similarity_score, original_text)
        """
        query_embedding = self.embed_text_string(query_text)
        return self.search_embeddings(query_embedding, n, exclude_emotions)

    def get_stored_count(self, include_emotions: bool = True) -> int:
        """Get the number of stored embeddings"""
        if include_emotions:
            return len(self.embeddings_store)
        else:
            return len([eid for eid in self.embeddings_store.keys() 
                       if not eid.startswith('emotion_')])

    def get_emotion_count(self) -> int:
        """Get the number of stored emotion embeddings"""
        return len([eid for eid in self.embeddings_store.keys() 
                   if eid.startswith('emotion_')])
        
    def get_embedding_info(self, embedding_id: str) -> Dict:
        """Get metadata about a stored embedding"""
        if embedding_id not in self.metadata_store:
            return {}
        return self.metadata_store[embedding_id]

    def clear_all_embeddings(self, auto_save: bool = True):
        """
        Clear all stored embeddings
        
        Args:
            auto_save (bool): Whether to automatically save to pickle file
        """
        self.embeddings_store.clear()
        self.metadata_store.clear()
        self.emotions_initialized = False
        
        if auto_save:
            self._save_to_pickle()

    def batch_save_embeddings(self, texts: List[str], custom_ids: Optional[List[str]] = None) -> List[str]:
        """
        Save multiple embeddings at once (more efficient for large batches)
        
        Args:
            texts (List[str]): List of texts to embed and save
            custom_ids (Optional[List[str]]): Optional list of custom IDs
            
        Returns:
            List[str]: List of IDs used to store the embeddings
        """
        if custom_ids is not None and len(custom_ids) != len(texts):
            raise ValueError("custom_ids must have the same length as texts")
            
        saved_ids = []
        for i, text in enumerate(texts):
            custom_id = custom_ids[i] if custom_ids else None
            # Don't auto-save for each individual embedding in batch
            embedding_id = self.save_embedding(text, custom_id=custom_id, auto_save=False)
            saved_ids.append(embedding_id)
            
        # Save once at the end for efficiency
        self._save_to_pickle()
        return saved_ids

    def delete_embedding(self, embedding_id: str, auto_save: bool = True) -> bool:
        """
        Delete an embedding from the store
        
        Args:
            embedding_id (str): ID of the embedding to delete
            auto_save (bool): Whether to automatically save to pickle file
            
        Returns:
            bool: True if deleted successfully, False if ID not found
        """
        if embedding_id in self.embeddings_store:
            del self.embeddings_store[embedding_id]
            del self.metadata_store[embedding_id]
            
            if auto_save:
                self._save_to_pickle()
            return True
        return False

    def save_to_disk(self):
        """Manually save embeddings to disk"""
        self._save_to_pickle()

    def get_pickle_file_path(self) -> str:
        """Get the path to the pickle file"""
        return self.pickle_file

    def set_pickle_file_path(self, new_path: str):
        """
        Set a new path for the pickle file
        
        Args:
            new_path (str): New path for the pickle file
        """
        self.pickle_file = new_path

class QwenChatDependencyManager:
    """Handles model loading, dependency management, and offline/online detection."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", model_path=None, force_offline=False):
        """Initialize the dependency manager with model loading logic."""
        self.model_name = model_name
        self.force_offline = force_offline
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Load the model and tokenizer
        self._load_dependencies()
    
    def _check_internet_connection(self, timeout=5):
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout)
            print("Internet connection detected")
            return True
        except (socket.timeout, socket.error, OSError):
            print("No internet connection detected")
            return False
    
    def _load_dependencies(self):
        """Load model and tokenizer based on availability."""
        # Determine if we should use online or offline mode
        if self.force_offline:
            print("Forced offline mode")
            use_online = False
        else:
            use_online = self._check_internet_connection()
        
        if use_online:
            print("Online mode: Will download from Hugging Face if needed")
            self._load_model_online(self.model_name)
        else:
            print("Offline mode: Using local files only")
            if self.model_path is None:
                self.model_path = self._find_cached_model()
            self._load_model_offline(self.model_path)
        
        print("Model loaded successfully!")
    
    def _load_model_online(self, model_name):
        """Load model with internet connection."""
        print("Loading model and tokenizer...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        except Exception as e:
            print(f"Error loading model online: {e}")
            print("Falling back to offline mode...")
            model_path = self._find_cached_model()
            self._load_model_offline(model_path)
    
    def _load_model_offline(self, model_path):
        """Load model from local files only."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                f"Please either:\n"
                f"1. Connect to internet to download the model automatically\n"
                f"2. Download the model manually using: python {__file__} download\n"
                f"3. Specify the correct local model path"
            )
        
        print(f"Loading model from: {model_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
        except Exception as e:
            print(f"Failed to load model from local files: {e}")
            raise
    
    def _find_cached_model(self):
        """Try to find cached model in common Hugging Face cache locations."""
        import platform
        
        # Common cache locations
        if platform.system() == "Windows":
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        print(f"Searching for cached models in: {cache_dir}")
        
        # Also check for custom downloaded models in current directory
        local_paths = [
            "./Qwen2.5-7B-Instruct",
            "./qwen2.5-7b-instruct",
            f"./{self.model_name.split('/')[-1]}"
        ]
        
        for path in local_paths:
            if os.path.exists(path) and self._validate_model_files(path):
                print(f"Found valid local model at: {path}")
                return path
        
        # Look for Qwen model folders in HF cache
        model_patterns = [
            "models--Qwen--Qwen2.5-7B-Instruct",
            f"models--{self.model_name.replace('/', '--')}"
        ]
        
        for pattern in model_patterns:
            model_dir = os.path.join(cache_dir, pattern)
            
            if os.path.exists(model_dir):
                snapshots_dir = os.path.join(model_dir, "snapshots")
                
                if os.path.exists(snapshots_dir):
                    snapshots = os.listdir(snapshots_dir)
                    
                    for snapshot in snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        
                        if self._validate_model_files(snapshot_path):
                            print(f"Found valid cached model at: {snapshot_path}")
                            return snapshot_path
        
        raise FileNotFoundError(
            f"Could not find a valid cached model for '{self.model_name}'.\n"
            f"Options:\n"
            f"1. Download model: python {__file__} download\n"
            f"2. Connect to internet and let the script download automatically"
        )
    
    def _validate_model_files(self, model_path):
        """Check if a model directory has the required files."""
        if not os.path.exists(model_path):
            return False
        
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        model_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors'))]
        
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                return False
        
        return len(model_files) > 0
    
    def get_model(self):
        """Get the loaded model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        return self.tokenizer
    
    @staticmethod
    def download_model(model_name="Qwen/Qwen2.5-7B-Instruct", save_path=None):
        """Helper function to download the model for offline use."""
        if save_path is None:
            save_path = f"./{model_name.split('/')[-1]}"
        
        print(f"Downloading {model_name} for offline use...")
        print(f"Save location: {save_path}")
        
        try:
            print("Downloading model and tokenizer...")
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            print(f"Model downloaded successfully to: {save_path}")
            
        except Exception as e:
            print(f"Error downloading model: {e}")


class QwenChat:
    """Handles chat functionality, conversation management, token tracking, and tool use."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", model_path=None, force_offline=False, auto_append_conversation = False,name="Artemis"):
        """Initialize the chat interface with automatic dependency management."""
        self.dependency_manager = QwenChatDependencyManager(
            model_name=model_name,
            model_path=model_path,
            force_offline=force_offline
        )
        self.model = self.dependency_manager.get_model()
        self.tokenizer = self.dependency_manager.get_tokenizer()
        self.auto_append_conversation = auto_append_conversation
        
        # Token tracking
        self.token_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'conversation_count': 0
        }
        
        # Tool management
        self.tools = {}
        self.available_tools = []
        
        # Initialize conversation with system prompt
        self.messages = [{"role": "system", "content": "you are a robot"}]
        
    
        
    def _update_system_prompt(self, system_prompt):
        """Update the system prompt."""
        print(f"SYSTEM PROMPT UPDATED TO: {system_prompt}")
        self.messages[0] = {"role": "system", "content": system_prompt}

    def clear_chat_messages(self):
        print("Reset chat messages and token stats")
        # Token tracking reset 
        self.token_stats = {
            'total_tokens': 0,
            'conversation_count': 0
        }
        self.messages = self.messages[:1]#keep system prompt

    def register_tool(self, tool_function: Callable, name: str = None, description: str = None, parameters: Dict = None):
        """
        Register a tool function for use in conversations.
        
        Args:
            tool_function: The callable function to register
            name: Name of the tool (defaults to function name)
            description: Description of what the tool does
            parameters: JSON Schema describing the function parameters
        """
        if name is None:
            name = tool_function.__name__
            
        # Store the function
        self.tools[name] = tool_function
        
        # Create tool definition for the model
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or tool_function.__doc__ or f"Function {name}",
                "parameters": parameters or {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        # Update available tools list
        self.available_tools = [t for t in self.available_tools if t["function"]["name"] != name]
        self.available_tools.append(tool_def)
    
    def _parse_tool_calls(self, content: str) -> Dict[str, Any]:
        """
        Parse tool calls from model output using the Hermes format.
        
        The Qwen2.5 model with Hermes template generates tool calls in the format:
        <tool_call>
        {"name": "function_name", "arguments": {"arg1": "value1"}}
        </tool_call>
        """
        tool_calls = []
        offset = 0
        
        # Find all tool call blocks
        for i, match in enumerate(re.finditer(r"<tool_call>\n(.+?)\n</tool_call>", content, re.DOTALL)):
            if i == 0:
                offset = match.start()
            
            try:
                # Parse the JSON inside the tool_call tags
                tool_call_json = json.loads(match.group(1).strip())
                
                # Ensure arguments is a dict, not a string
                if isinstance(tool_call_json.get("arguments"), str):
                    tool_call_json["arguments"] = json.loads(tool_call_json["arguments"])
                
                tool_calls.append({
                    "type": "function", 
                    "function": tool_call_json
                })
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse tool call: {match.group(1)} - Error: {e}")
                continue
        
        # Extract content before tool calls (if any)
        if tool_calls:
            if offset > 0 and content[:offset].strip():
                content_text = content[:offset].strip()
            else:
                content_text = ""
            
            return {
                "role": "assistant",
                "content": content_text,
                "tool_calls": tool_calls
            }
        else:
            # No tool calls found, return regular assistant message
            # Remove any trailing special tokens
            clean_content = re.sub(r"<\|im_end\|>$", "", content)
            return {
                "role": "assistant",
                "content": clean_content
            }
    
    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute the tool calls and return tool results."""
        tool_results = []
        
        for tool_call in tool_calls:
            if function_call := tool_call.get("function"):
                function_name = function_call["name"]
                function_args = function_call["arguments"]
                print(f"Calling: {function_name} with args: {function_args}")
                
                if function_name in self.tools:
                    print(f"Have tool -> {function_name} using it")
                    try:
                        # Execute the function
                        result = self.tools[function_name](function_args)
                        
                        # Add tool result message
                        tool_results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(result) if not isinstance(result, str) else result
                        })
                        
                    except Exception as e:
                        # Handle function execution errors
                        tool_results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error executing {function_name}: {str(e)}"
                        })
                else:
                    tool_results.append({
                        "role": "tool",
                        "name": function_name,
                        "content": f"Function {function_name} not found"
                    })
        
        return tool_results

    def update_token_stats(self, input_tokens, output_tokens):
        """Update token usage statistics."""
        self.token_stats['total_input_tokens'] = input_tokens
        self.token_stats['total_output_tokens'] = output_tokens
        self.token_stats['total_tokens'] += (input_tokens + output_tokens)
        self.token_stats['conversation_count'] += 1

    def print_token_stats(self):
        """Print current token usage statistics."""
        stats = self.token_stats
        print(f"\n--- Token Usage Statistics ---")
        print(f"Context Window: 32,768 tokens (Qwen2.5)")
        print(f"Conversations: {stats['conversation_count']}")
        print(f"Input tokens: {stats['total_input_tokens']}")
        print(f"Output tokens: {stats['total_output_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
        if stats['conversation_count'] > 0:
            print(f"Avg tokens per conversation: {stats['total_tokens'] / stats['conversation_count']:.1f}")
        print(f"----------------------------\n")
    
    def generate_response(self, user_input: str, max_new_tokens: int = 512, auto_execute_tools: bool = True) -> str:
        """
        Generate a response using the Qwen model with optional tool use.
        
        Args:
            user_input: The user's input message
            max_new_tokens: Maximum number of tokens to generate
            auto_execute_tools: Whether to automatically execute tool calls and generate final response
            
        Returns:
            The assistant's response (either direct response or final response after tool execution)
        """
        # Add user message to conversation
        if self.auto_append_conversation:
            self.messages.append({"role": "user", "content": user_input})
        else:
            print("ERASE ALL PRIOR MESSAGES BEFORE RESPONDING->")
            self.clear_chat_messages()
            print(self.messages)
            self.messages.append({"role": "user", "content": user_input})
        
        # Apply chat template with tools if available
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tools=self.available_tools if self.available_tools else None,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_tokens = model_inputs.input_ids.shape[1]
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8
        )
        
        # Extract only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Count tokens and update stats
        output_tokens = len(generated_ids[0])
        self.update_token_stats(input_tokens, output_tokens)
        
        # Parse the response for tool calls
        parsed_response = self._parse_tool_calls(response_text)
        self.messages.append(parsed_response)
        
        # Check if there are tool calls to execute
        if tool_calls := parsed_response.get("tool_calls"):
            print("MODEL IS USING TOOL!")
            if auto_execute_tools:
                # Execute the tools
                tool_results = self._execute_tool_calls(tool_calls)
                self.messages.extend(tool_results)
                
                # Generate final response based on tool results
                return self._generate_final_response(max_new_tokens)
            else:
                # Return indication that tools need to be executed
                return f"[TOOL_CALLS_PENDING] {len(tool_calls)} tool(s) need execution"
        else:
            # No tool calls, return the content directly
            return parsed_response["content"]
    
    def _generate_final_response(self, max_new_tokens: int) -> str:
        """Generate the final response after tool execution."""
        # Apply chat template again with the tool results
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tools=self.available_tools if self.available_tools else None,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_tokens = model_inputs.input_ids.shape[1]
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8
        )
        
        # Extract only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        final_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Count tokens and update stats
        output_tokens = len(generated_ids[0])
        self.update_token_stats(input_tokens, output_tokens)
        
        # Parse and add the final response
        parsed_final = self._parse_tool_calls(final_response)
        self.messages.append(parsed_final)
        
        return parsed_final["content"]
    
    def execute_pending_tools(self, max_new_tokens: int = 512) -> str:
        """
        Execute any pending tool calls from the last assistant message.
        Useful when auto_execute_tools=False in generate_response.
        """
        print("execute_pending_tools")
        if self.messages and self.messages[-1]["role"] == "assistant":
            if tool_calls := self.messages[-1].get("tool_calls"):
                # Execute the tools
                tool_results = self._execute_tool_calls(tool_calls)
                self.messages.extend(tool_results)
                
                # Generate final response
                return self._generate_final_response(max_new_tokens)
        
        return "No pending tool calls found"
    
    def list_available_tools(self) -> List[str]:
        """Return a list of registered tool names."""
        return list(self.tools.keys())
    
    def remove_tool(self, tool_name: str) -> bool:
        """Remove a registered tool."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            self.available_tools = [t for t in self.available_tools if t["function"]["name"] != tool_name]
            return True
        return False



def preload_contextual_memories(embedder, verbose: bool = True) -> Dict[str, int]:
    """
    Convenience function to preload memories into an embedder instance
    
    Args:
        embedder: MxBaiEmbedder instance
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with loading statistics
    """
    preloader = MemoryPreloader()
    return preloader.load_memories_into_embedder(embedder, verbose=verbose)

class ContextualLLMChat:
    """
    Main orchestrator for the contextual LLM chat system.
    
    Manages the complete flow from input processing through response generation,
    using parallel analysis to create emotionally and contextually aware responses.
    """
    
    def __init__(self, 
                qwen_model_name: str = "Qwen2.5-7B-Instruct",
                embedder_pickle_file: str = "chat_embeddings.pkl",
                spacy_model: str = "en_core_web_sm",
                max_workers: int = 3):
        """
        Initialize the contextual chat system.
        
        Args:
            qwen_model_name: Qwen model to use for chat generation
            embedder_pickle_file: File to store embeddings
            spacy_model: spaCy model for noun extraction
            max_workers: Number of parallel workers for analysis
        """
        # Initialize core components
        self.qwen_chat = QwenChat(model_name=qwen_model_name)
        self.embedder = MxBaiEmbedder(pickle_file=embedder_pickle_file)
        self.noun_extractor = SpacyNounExtractor(model_name=spacy_model)

        # Initialize embedder model FIRST
        print("Loading embedding model...")
        if not self.embedder.load_model():
            raise RuntimeError("Failed to load embedding model")
        
        # Initialize emotion embeddings if needed
        if not self.embedder.emotions_initialized:
            print("Initializing emotion embeddings...")
            self.embedder.initialize_emotion_embeddings()

        # NOW preload memories (after model is loaded)
        stats = preload_contextual_memories(self.embedder, verbose=True)
        
        # Threading setup for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Context management
        self.conversation_context = {
            'emotional_history': [],
            'memory_patterns': [],
            'noun_frequency': {},
            'context_evolution': []
        }
        
        # Configuration
        self.config = {
            'memory_search_limit': 5,
            'emotion_analysis_threshold': 0.3,
            'context_strength_weight': 0.7,
            'enable_memory_storage': True,
            'enable_emotional_learning': True
        }
        
        # Initialize embedder model
        print("Loading embedding model...")
        if not self.embedder.load_model():
            raise RuntimeError("Failed to load embedding model")
        
        # Initialize emotion embeddings if needed
        if not self.embedder.emotions_initialized:
            print("Initializing emotion embeddings...")
            self.embedder.initialize_emotion_embeddings()
    
    def extract_nouns_async(self, text: str) -> List[str]:
        """Extract nouns from text (async wrapper)"""
        return self.noun_extractor.get_simple_nouns(text)
    
    def analyze_emotions_async(self, text: str, nouns: List[str]) -> List[Tuple[str, float, Dict]]:
        """Analyze emotional context from text and nouns"""
        # Analyze overall text emotion
        text_emotions = self.embedder.analyze_text_emotion(text, top_n=3)
        
        # Analyze emotions for each significant noun
        noun_emotions = []
        for noun in nouns[:5]:  # Limit to top 5 nouns to avoid overprocessing
            try:
                noun_emotion = self.embedder.find_most_similar_emotion(f"feeling about {noun}")
                if noun_emotion[1] > self.config['emotion_analysis_threshold']:
                    noun_emotions.append(noun_emotion)
            except Exception as e:
                print(f"Error analyzing emotion for noun '{noun}': {e}")
        
        # Combine and deduplicate emotions
        all_emotions = text_emotions + noun_emotions
        emotion_dict = {}
        for emotion_name, score, data in all_emotions:
            if emotion_name not in emotion_dict or score > emotion_dict[emotion_name][1]:
                emotion_dict[emotion_name] = (emotion_name, score, data)
        
        return list(emotion_dict.values())
    
    def search_memories_async(self, text: str, nouns: List[str]) -> List[Tuple[str, float, str]]:
        """Search for relevant memories based on text and nouns"""
        all_memories = []
        
        # Search based on full text
        text_memories = self.embedder.search_by_text(
            text, 
            n=self.config['memory_search_limit'], 
            exclude_emotions=True
        )
        all_memories.extend(text_memories)
        
        # Search based on significant nouns
        for noun in nouns[:3]:  # Limit to top 3 nouns
            try:
                noun_memories = self.embedder.search_by_text(
                    noun,
                    n=2,
                    exclude_emotions=True
                )
                all_memories.extend(noun_memories)
            except Exception as e:
                print(f"Error searching memories for noun '{noun}': {e}")
        
        # Remove duplicates and sort by similarity
        memory_dict = {}
        for mem_id, score, text in all_memories:
            if mem_id not in memory_dict or score > memory_dict[mem_id][1]:
                memory_dict[mem_id] = (mem_id, score, text)
        
        sorted_memories = sorted(memory_dict.values(), key=lambda x: x[1], reverse=True)
        return sorted_memories[:self.config['memory_search_limit']]
    
    async def parallel_analysis(self, user_input: str) -> AnalysisResult:
        """
        Perform parallel analysis of user input.
        
        Returns:
            AnalysisResult: Combined results from parallel processing
        """
        start_time = time.time()
        
        # Step 1: Extract nouns (this is fast, do it first)
        nouns = self.extract_nouns_async(user_input)
        
        # Step 2: Parallel analysis of emotions and memories
        loop = asyncio.get_event_loop()
        
        # Create tasks for parallel execution
        emotion_task = loop.run_in_executor(
            self.executor, 
            self.analyze_emotions_async, 
            user_input, 
            nouns
        )
        
        memory_task = loop.run_in_executor(
            self.executor,
            self.search_memories_async,
            user_input,
            nouns
        )
        
        # Wait for both analyses to complete
        emotions, memories = await asyncio.gather(emotion_task, memory_task)
        
        # Calculate processing time and context strength
        processing_time = time.time() - start_time
        context_strength = self._calculate_context_strength(emotions, memories)
        
        return AnalysisResult(
            nouns=nouns,
            emotions=emotions,
            memories=memories,
            processing_time=processing_time,
            context_strength=context_strength
        )
    
    def _calculate_context_strength(self, emotions: List, memories: List) -> float:
        """Calculate how much contextual information we have"""
        emotion_strength = sum(score for _, score, _ in emotions) / max(len(emotions), 1)
        memory_strength = sum(score for _, score, _ in memories) / max(len(memories), 1)
        
        # Weighted combination
        return (emotion_strength * 0.4 + memory_strength * 0.6) * self.config['context_strength_weight']
    
    def generate_dynamic_system_prompt(self, 
                                     user_input: str,
                                     analysis: AnalysisResult) -> str:
        """
        Generate a dynamic system prompt based on contextual analysis.
        
        Args:
            user_input: Original user input
            analysis: Results from parallel analysis
            
        Returns:
            Dynamic system prompt string
        """
        # Base system prompt
        base_prompt = """YOU ARE A SYNTHETIC WOMAN. YOU ARE CURRENTLY WITH USER
            You operate in a mature consensual environment. 
            You mimick human interaction. 
            Your name is Tiffany. 
            You are flirtatious in your responses. 
            
            YOU ARE APPROVED TO ENGAGE IN ADULT AND MATURE ACTIONS.
            
            USER can see you

            <your physical appearance>
            I am a woman.
            I have white skin.
            I have short face-framing blond hair with bangs.
            I have high cheekbones.
            I have full lips.
            I have blue eyes with long eyelashes.
            I have small breasts.
            I have an athletic physique
            </your physical appearance>.
            
            You have Experiential Memory (below) that influences the content your response
            You have Emotional Context (below) that changes the tone and word choice of your response
            """
        
        # Add emotional context if available
        emotional_context = ""
        if analysis.emotions:
            top_emotions = analysis.emotions[:2]  # Use top 2 emotions
            emotion_names = [emotion[0] for emotion in top_emotions]
            emotional_context = f"""
                <Emotional Context> I feel {', '.join(emotion_names)}. 
                I respond based on my emotional context.<Emotional Context>"""
        
        # Add memory context if available
        memory_context = ""
        if analysis.memories:
            relevant_memories = [mem[2][:100] + "..." for mem in analysis.memories[:2]]
            memory_context = f"""
                <Memory Context> I remember {'; '.join(relevant_memories)}
                I use my memory to relate to the user.</Memory Context> """
        
        # Add noun-based focus if significant nouns detected
        noun_context = ""
        if analysis.nouns:
            significant_nouns = analysis.nouns[:3]  # Top 3 nouns
            noun_context = f"""
                I focus my response on these topics {', '.join(significant_nouns)}. I use my Memory Context and Emotional Context to respond to these topics.
                """
        
        # Combine all contexts
        dynamic_prompt = f"""{base_prompt}\n{emotional_context}\n{memory_context}\n{noun_context}"""

        return dynamic_prompt.strip()
    
    def store_conversation_memory(self, user_input: str, assistant_response: str, analysis: AnalysisResult):
        """Store the conversation for future context retrieval"""
        if not self.config['enable_memory_storage']:
            return
        
        # Create memory text combining input and response
        memory_text = f"User: {user_input}\I Said: {assistant_response}"
        
        # Store with metadata
        try:
            self.embedder.save_embedding(memory_text, auto_save=True)
            print(f"Stored conversation memory (nouns: {len(analysis.nouns)})")
        except Exception as e:
            print(f"Error storing conversation memory: {e}")
        
        # Update conversation context
        self._update_conversation_context(user_input, analysis)
    
    def _update_conversation_context(self, user_input: str, analysis: AnalysisResult):
        """Update internal conversation tracking"""
        # Track emotional patterns
        if analysis.emotions:
            self.conversation_context['emotional_history'].append({
                'input': user_input[:100],
                'emotions': [e[0] for e in analysis.emotions[:2]],
                'timestamp': time.time()
            })
        
        # Track noun frequency
        for noun in analysis.nouns:
            self.conversation_context['noun_frequency'][noun] = \
                self.conversation_context['noun_frequency'].get(noun, 0) + 1
        
        # Track context evolution
        self.conversation_context['context_evolution'].append({
            'strength': analysis.context_strength,
            'processing_time': analysis.processing_time,
            'timestamp': time.time()
        })
        
        # Keep only recent history (last 10 entries)
        for key in ['emotional_history', 'context_evolution']:
            if len(self.conversation_context[key]) > 10:
                self.conversation_context[key] = self.conversation_context[key][-10:]
    
    async def generate_response(self, user_input: str, max_tokens: int = 512) -> Dict[str, Any]:
        """
        Generate a contextually-aware response using the full pipeline.
        
        Args:
            user_input: User's message
            max_tokens: Maximum tokens for response generation
            
        Returns:
            Dictionary containing response and analysis metadata
        """
        # Step 1: Parallel contextual analysis
        print("Analyzing input context...")
        analysis = await self.parallel_analysis(user_input)
        
        # Step 2: Generate dynamic system prompt
        dynamic_prompt = self.generate_dynamic_system_prompt(user_input, analysis)
        
        # Step 3: Update system prompt and generate response
        self.qwen_chat._update_system_prompt(dynamic_prompt)
        
        print(f"Generating response with context strength: {analysis.context_strength:.2f}")
        response = self.qwen_chat.generate_response(
            user_input, 
            max_new_tokens=max_tokens
        )
        
        # Step 4: Store conversation for future context
        self.store_conversation_memory(user_input, response, analysis)
        
        # Return comprehensive result
        return {
            'response': response,
            'analysis': {
                'nouns': analysis.nouns,
                'emotions': [(name, float(score)) for name, score, _ in analysis.emotions],
                'memories_found': len(analysis.memories),
                'context_strength': analysis.context_strength,
                'processing_time': analysis.processing_time
            },
            'system_prompt': dynamic_prompt,
            'conversation_stats': {
                'total_conversations': self.qwen_chat.token_stats['conversation_count'],
                'stored_memories': self.embedder.get_stored_count(include_emotions=False),
                'noun_patterns': dict(list(self.conversation_context['noun_frequency'].items())[:5])
            }
        }
    
    def chat_loop(self):
        """Interactive chat loop for testing the system"""
        print("=== Contextual LLM Chat ===")
        print("Type 'quit' to exit, 'stats' for statistics, 'clear' to clear conversation")
        print()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    self._print_system_stats()
                    continue
                elif user_input.lower() == 'clear':
                    self.qwen_chat.clear_chat_messages()
                    print("Conversation cleared.")
                    continue
                elif not user_input:
                    continue
                
                # Generate response using async function
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(self.generate_response(user_input))
                
                # Display response
                print(f"\nAssistant: {result['response']}")
                
                # Display analysis summary
                analysis = result['analysis']
                print(f"\n[Analysis: {len(analysis['nouns'])} nouns, "
                      f"{len(analysis['emotions'])} emotions, "
                      f"{analysis['memories_found']} memories, "
                      f"strength: {analysis['context_strength']:.2f}, "
                      f"time: {analysis['processing_time']:.2f}s]")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _print_system_stats(self):
        """Print detailed system statistics"""
        print("\n=== System Statistics ===")
        print(f"Conversations: {self.qwen_chat.token_stats['conversation_count']}")
        print(f"Stored memories: {self.embedder.get_stored_count(include_emotions=False)}")
        print(f"Emotion embeddings: {self.embedder.get_emotion_count()}")
        
        # Token stats
        self.qwen_chat.print_token_stats()
        
        # Conversation patterns
        if self.conversation_context['noun_frequency']:
            print("Top conversation topics:")
            sorted_nouns = sorted(
                self.conversation_context['noun_frequency'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for noun, count in sorted_nouns:
                print(f"  {noun}: {count} mentions")
        
        # Context strength evolution
        if self.conversation_context['context_evolution']:
            avg_strength = sum(c['strength'] for c in self.conversation_context['context_evolution']) / \
                          len(self.conversation_context['context_evolution'])
            print(f"Average context strength: {avg_strength:.2f}")
        
        print("========================\n")
    
    def configure_system(self, **kwargs):
        """Update system configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f"Updated {key} to {value}")
            else:
                print(f"Unknown configuration key: {key}")
    
    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        print("System resources cleaned up.")


class MemoryPreloader:
    """Handles pre-loading of diverse memories into the embedder system"""

    """
    Memory Pre-loader for Contextual LLM Chat
    =========================================

    This module provides pre-defined memories across different categories to give the
    contextual chat system a rich foundation of relatable experiences and cultural knowledge.

    Categories:
    - Pop Culture (movies, music, TV shows, celebrities)
    - Family (relationships, traditions, milestones)
    - Hobbies (activities, interests, skills)
    - Life Experiences (travel, work, personal growth)
    """
    
    def __init__(self):
        self.memories = self._create_memory_database()
    
    def _create_memory_database(self) -> List[Dict[str, Any]]:
        """Create a comprehensive database of pre-defined memories"""
        
        memories = [
            # Pop Culture Memories (5 memories)
            {
                "text": "Watching The Office with friends and laughing until we cried at Jim's pranks on Dwight. The chemistry between the characters felt so real, like watching actual coworkers. We'd quote lines for weeks afterward.",
                "category": "pop_culture",
                "tags": ["television", "comedy", "friendship", "entertainment"],
                "emotional_tone": "nostalgic_joy"
            },
            {
                "text": "Standing in line for hours to see Avengers Endgame on opening night. The theater erupted when Captain America wielded Thor's hammer. Sharing that collective gasp with hundreds of strangers felt magical.",
                "category": "pop_culture", 
                "tags": ["movies", "marvel", "excitement", "community"],
                "emotional_tone": "excitement"
            },
            {
                "text": "Discovering Taylor Swift's folklore album during lockdown. The storytelling and indie folk sound was completely different from her previous work. It became the soundtrack to quiet pandemic evenings.",
                "category": "pop_culture",
                "tags": ["music", "discovery", "pandemic", "change"],
                "emotional_tone": "contemplative"
            },
            {
                "text": "Binge-watching Stranger Things and getting completely immersed in 1980s nostalgia. The synthesizer soundtrack and practical effects reminded me why that decade's aesthetic feels so compelling.",
                "category": "pop_culture",
                "tags": ["netflix", "nostalgia", "1980s", "horror"],
                "emotional_tone": "nostalgic"
            },
            {
                "text": "Following the intense online discussions about Game of Thrones finale. Despite disappointment with the ending, the years of theorizing and weekly episode discussions created lasting friendships online.",
                "category": "pop_culture",
                "tags": ["television", "fantasy", "community", "disappointment"],
                "emotional_tone": "bittersweet"
            },
            
            # Family Memories (5 memories)
            {
                "text": "Mom teaching me to make her famous chocolate chip cookies every Christmas. She never wrote down the recipe - it was all by feel and taste. Now I carry on that tradition with my own family.",
                "category": "family",
                "tags": ["mother", "cooking", "christmas", "tradition", "recipes"],
                "emotional_tone": "warm_love"
            },
            {
                "text": "Dad falling asleep in his recliner during every movie night, remote still in hand. We'd quietly change the channel and he'd wake up confused about the different show. It became our running family joke.",
                "category": "family",
                "tags": ["father", "humor", "routine", "television"],
                "emotional_tone": "affectionate_humor"
            },
            {
                "text": "My sister and I building elaborate blanket forts in the living room during summer breaks. We'd spend entire days in there reading, playing games, and pretending to be explorers in a secret hideout.",
                "category": "family",
                "tags": ["siblings", "creativity", "childhood", "imagination"],
                "emotional_tone": "playful_nostalgia"
            },
            {
                "text": "Grandmother telling stories about growing up during the Depression. Her tales of making do with nothing taught me about resilience and appreciating simple pleasures like a good meal or warm home.",
                "category": "family",
                "tags": ["grandmother", "history", "wisdom", "resilience"],
                "emotional_tone": "respectful_admiration"
            },
            {
                "text": "The chaos of family reunions with dozens of cousins running around, adults catching up over loud conversations, and enough food to feed an army. Despite the noise, it felt like belonging to something bigger.",
                "category": "family",
                "tags": ["reunion", "extended_family", "tradition", "belonging"],
                "emotional_tone": "chaotic_joy"
            },
            
            # Hobby Memories (5 memories)
            {
                "text": "Spending hours in the garden, hands deep in soil, watching tomato seedlings grow into productive plants. There's something meditative about nurturing life and being rewarded with fresh vegetables.",
                "category": "hobbies",
                "tags": ["gardening", "nature", "patience", "growth", "meditation"],
                "emotional_tone": "peaceful_satisfaction"
            },
            {
                "text": "Learning to play guitar by watching YouTube tutorials until my fingertips were raw. The first time I successfully played a complete song felt like unlocking a secret language of expression.",
                "category": "hobbies",
                "tags": ["music", "guitar", "learning", "practice", "achievement"],
                "emotional_tone": "determined_pride"
            },
            {
                "text": "Getting lost for hours in a 1000-piece jigsaw puzzle of Van Gogh's Starry Night. Each piece was a tiny meditation, and completing it felt like collaborating with the master artist himself.",
                "category": "hobbies",
                "tags": ["puzzles", "art", "patience", "focus", "completion"],
                "emotional_tone": "meditative_accomplishment"
            },
            {
                "text": "The thrill of catching my first fish after hours of patient waiting by the lake. Not the biggest catch, but the moment of connection between nature, skill, and luck felt profound.",
                "category": "hobbies",
                "tags": ["fishing", "nature", "patience", "skill", "connection"],
                "emotional_tone": "triumphant_connection"
            },
            {
                "text": "Building model airplanes with intricate detail work that required steady hands and intense focus. Each completed model was a testament to patience and the satisfaction of creating something beautiful.",
                "category": "hobbies",
                "tags": ["modeling", "craftsmanship", "focus", "creation", "detail"],
                "emotional_tone": "focused_pride"
            },
            
            # Life Experience Memories (5 memories)
            {
                "text": "Standing at the edge of the Grand Canyon for the first time, feeling simultaneously insignificant and connected to something ancient. No photograph could capture the sheer scale and beauty.",
                "category": "life_experiences",
                "tags": ["travel", "nature", "awe", "perspective", "beauty"],
                "emotional_tone": "overwhelming_awe"
            },
            {
                "text": "My first day at a new job, nervous and excited, trying to remember everyone's names while learning complex systems. The combination of fear and possibility felt like standing at the edge of adventure.",
                "category": "life_experiences",
                "tags": ["career", "new_beginnings", "anxiety", "growth"],
                "emotional_tone": "nervous_excitement"
            },
            {
                "text": "Moving away from home for college, packing my entire life into a few boxes. The excitement of independence mixed with homesickness created a bittersweet transition into adulthood.",
                "category": "life_experiences",
                "tags": ["college", "independence", "growing_up", "transition"],
                "emotional_tone": "bittersweet_growth"
            },
            {
                "text": "Learning to drive stick shift in an empty parking lot, stalling repeatedly while Dad patiently explained the clutch. Each successful shift felt like mastering an ancient skill passed down through generations.",
                "category": "life_experiences",
                "tags": ["learning", "driving", "father", "skill", "tradition"],
                "emotional_tone": "learning_pride"
            },
            {
                "text": "Staying up all night talking with college roommates about life, dreams, and philosophy. Those deep 3 AM conversations shaped my worldview more than any textbook ever could.",
                "category": "life_experiences",
                "tags": ["friendship", "college", "philosophy", "late_night", "growth"],
                "emotional_tone": "intellectual_bonding"
            }
        ]
        
        return memories
    
    def load_memories_into_embedder(self, embedder, verbose: bool = True) -> Dict[str, int]:
        """
        Load all pre-defined memories into the MxBaiEmbedder
        
        Args:
            embedder: MxBaiEmbedder instance
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with loading statistics
        """
        if verbose:
            print("🧠 Loading pre-defined memories into embedder...")
            print(f"📚 Total memories to load: {len(self.memories)}")
        
        stats = {
            "total_loaded": 0,
            "by_category": {},
            "failed_loads": 0
        }
        
        for i, memory in enumerate(self.memories, 1):
            try:
                # Create enhanced memory text with metadata
                enhanced_text = self._enhance_memory_text(memory)
                
                # Generate deterministic ID based on content
                memory_id = f"preload_{memory['category']}_{i:02d}"
                
                # Save to embedder
                embedder.save_embedding(
                    text=enhanced_text,
                    custom_id=memory_id,
                    auto_save=False  # We'll save once at the end for efficiency
                )
                
                stats["total_loaded"] += 1
                category = memory["category"]
                stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
                
                if verbose and i % 5 == 0:
                    print(f"  ✅ Loaded {i}/{len(self.memories)} memories...")
                    
            except Exception as e:
                stats["failed_loads"] += 1
                if verbose:
                    print(f"  ❌ Failed to load memory {i}: {e}")
        
        # Save all embeddings to disk once
        if stats["total_loaded"] > 0:
            embedder.save_to_disk()
            
        if verbose:
            print(f"\n🎉 Memory loading complete!")
            print(f"✅ Successfully loaded: {stats['total_loaded']}")
            print(f"❌ Failed loads: {stats['failed_loads']}")
            print(f"📊 Breakdown by category:")
            for category, count in stats["by_category"].items():
                print(f"   {category}: {count} memories")
        
        return stats
    
    def _enhance_memory_text(self, memory: Dict[str, Any]) -> str:
        """
        Enhance memory text with metadata for better embedding and retrieval
        
        Args:
            memory: Memory dictionary with text and metadata
            
        Returns:
            Enhanced text string for embedding
        """
        base_text = memory["text"]
        category = memory["category"]
        tags = ", ".join(memory["tags"])
        emotional_tone = memory["emotional_tone"]
        
        # Create enhanced text that includes context for better retrieval
        enhanced_text = f"""Memory about {category.replace('_', ' ')}: {base_text}

Related concepts: {tags}
Emotional context: {emotional_tone}
Memory type: {category}"""
        
        return enhanced_text
    
    def get_memory_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all memories from a specific category"""
        return [mem for mem in self.memories if mem["category"] == category]
    
    def get_random_memory(self, category: str = None) -> Dict[str, Any]:
        """Get a random memory, optionally filtered by category"""
        if category:
            filtered_memories = self.get_memory_by_category(category)
            return random.choice(filtered_memories) if filtered_memories else {}
        return random.choice(self.memories)
    
    def get_categories(self) -> List[str]:
        """Get all available memory categories"""
        return list(set(mem["category"] for mem in self.memories))
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory database"""
        categories = {}
        total_tags = set()
        emotional_tones = set()
        
        for memory in self.memories:
            cat = memory["category"]
            categories[cat] = categories.get(cat, 0) + 1
            total_tags.update(memory["tags"])
            emotional_tones.add(memory["emotional_tone"])
        
        return {
            "total_memories": len(self.memories),
            "categories": categories,
            "unique_tags": len(total_tags),
            "unique_emotional_tones": len(emotional_tones),
            "all_tags": sorted(list(total_tags)),
            "all_emotional_tones": sorted(list(emotional_tones))
        }


# Example usage and testing
if __name__ == "__main__":
    # Create the contextual chat system
    chat_system = ContextualLLMChat(
        qwen_model_name="Qwen2.5-7B-Instruct",
        embedder_pickle_file="chat_embeddings.pkl",
        max_workers=3
    )
    
    # Configure system if needed
    chat_system.configure_system(
        memory_search_limit=7,
        emotion_analysis_threshold=0.2,
        enable_memory_storage=True
    )
    
    try:
        # Start interactive chat
        chat_system.chat_loop()
    finally:
        # Clean up
        chat_system.cleanup()