from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Type
import uuid
import time
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import math
import random
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import spacy


# Import our memory system components
from qwen3_emotion_memory import MxBaiEmbedder


# Pseudo embedding Classes - PLACE HOLDERS for actual implementations 
#audio
class pseudo_vggish_embed:
    def pseudo_vggish_embed(self,audio_file: str) -> np.ndarray:
        """
        Pseudo VGGish audio embedding function
        In reality, this would process audio files and return 128-dimensional embeddings
        """
        # Simulate VGGish 128-dimensional embedding
        np.random.seed(hash(audio_file) % 2**32)  # Deterministic based on filename
        embedding = np.random.normal(0, 1, 128)
        # Normalize like real VGGish embeddings
        embedding = embedding / np.linalg.norm(embedding)
        print(f"[VGGish] Generated audio embedding for {audio_file} (shape: {embedding.shape})")
        return embedding
#image
class pseudo_beit_embed:

    def pseudo_beit_embed(self,image_file: str) -> np.ndarray:
        """
        Pseudo BeiT-large-patch-22f image embedding function
        In reality, this would process images and return 1024-dimensional embeddings
        """
        # Simulate BeiT-large 1024-dimensional embedding
        np.random.seed(hash(image_file) % 2**32)  # Deterministic based on filename
        embedding = np.random.normal(0, 1, 1024)
        # Normalize like real vision transformer embeddings
        embedding = embedding / np.linalg.norm(embedding)
        print(f"[BeiT] Generated image embedding for {image_file} (shape: {embedding.shape})")
        return embedding


#emotional response
class EmotionEngine:
    """
    Emotion engine that generates emotional responses based on input text.
    This is a placeholder for a more complex emotional analysis system.
    """
    
    def __init__(self,embedding_model: Type[MxBaiEmbedder] = None):
        self.embedding_model = embedding_model() #MxBaiEmbedder()
        
        # Plutchik's primary emotions with their characteristics
        self.ee_emotion_database = {
            # Joy family
            'joy': {
                'mood': 'Sense of energy and possibility',
                'thoughts': 'Life is going well',
                'responses': 'Sparks creativity, connection, gives energy',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'ecstasy': {
                'mood': 'Overwhelming euphoria and elation',
                'thoughts': 'Everything is perfect and amazing',
                'responses': 'Boundless enthusiasm, may act impulsively from excitement',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'serenity': {
                'mood': 'Calm contentment and peace',
                'thoughts': 'Things are pleasant and stable',
                'responses': 'Gentle actions, seeks to maintain harmony',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            
            # Sadness family
            'sadness': {
                'mood': 'Heavy, low energy, withdrawn',
                'thoughts': 'Things aren\'t going well, feeling loss',
                'responses': 'Seeks comfort, may isolate, moves slowly',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'grief': {
                'mood': 'Profound sorrow and despair',
                'thoughts': 'Something important is gone forever',
                'responses': 'May be inconsolable, needs support, difficulty functioning',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'pensiveness': {
                'mood': 'Quiet melancholy and reflection',
                'thoughts': 'Contemplating what could have been',
                'responses': 'Introspective, seeks solitude, gentle sadness',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            
            # Trust family
            'trust': {
                'mood': 'Open and accepting',
                'thoughts': 'Others are reliable and good',
                'responses': 'Cooperative, shares freely, seeks connection',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'admiration': {
                'mood': 'Deep respect and reverence',
                'thoughts': 'This person/thing is truly worthy',
                'responses': 'Wants to learn, emulate, or serve',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'acceptance': {
                'mood': 'Calm acknowledgment',
                'thoughts': 'This is how things are',
                'responses': 'Goes with the flow, doesn\'t resist',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            
            # Disgust family
            'disgust': {
                'mood': 'Repulsed and rejecting',
                'thoughts': 'This is wrong, contaminated, or inferior',
                'responses': 'Avoids, criticizes, seeks to remove or cleanse',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'loathing': {
                'mood': 'Intense revulsion and hatred',
                'thoughts': 'This is absolutely abhorrent',
                'responses': 'Strong rejection, may become aggressive to eliminate',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'boredom': {
                'mood': 'Mild disinterest and restlessness',
                'thoughts': 'This isn\'t worth my attention',
                'responses': 'Seeks stimulation elsewhere, disengages',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            
            # Fear family
            'fear': {
                'mood': 'Anxious alertness and tension',
                'thoughts': 'Something bad might happen',
                'responses': 'Cautious, seeks safety, may freeze or flee',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'terror': {
                'mood': 'Paralyzing dread',
                'thoughts': 'Immediate danger, might not survive',
                'responses': 'Fight, flight, or freeze response, acts on instinct',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'apprehension': {
                'mood': 'Mild worry and uncertainty',
                'thoughts': 'Something doesn\'t feel quite right',
                'responses': 'More cautious than usual, seeks reassurance',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            
            # Anger family
            'anger': {
                'mood': 'Heated and energized',
                'thoughts': 'This is unfair, I\'ve been wronged',
                'responses': 'Confrontational, seeks to correct or punish',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'rage': {
                'mood': 'Burning fury and aggression',
                'thoughts': 'Must destroy the source of this injustice',
                'responses': 'Potentially violent, loses rational control',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'annoyance': {
                'mood': 'Mildly irritated and impatient',
                'thoughts': 'This is inconvenient or bothersome',
                'responses': 'Short responses, may express frustration verbally',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            
            # Surprise family
            'surprise': {
                'mood': 'Startled and alert',
                'thoughts': 'That was unexpected',
                'responses': 'Heightened attention, pauses to process',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'amazement': {
                'mood': 'Awed and wonder-struck',
                'thoughts': 'This is incredible and beyond belief',
                'responses': 'Stares, asks questions, wants to understand',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'distraction': {
                'mood': 'Mildly surprised and unfocused',
                'thoughts': 'Wait, what was that?',
                'responses': 'Attention shifts, momentarily loses focus',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            
            # Anticipation family
            'anticipation': {
                'mood': 'Eager and forward-looking',
                'thoughts': 'Something good is coming',
                'responses': 'Prepares, plans, may act impatiently',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'vigilance': {
                'mood': 'Intense focus and readiness',
                'thoughts': 'Must be ready for what\'s coming',
                'responses': 'Hyper-alert, prepared for action',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'interest': {
                'mood': 'Curious and engaged',
                'thoughts': 'I want to know more about this',
                'responses': 'Asks questions, explores, pays attention',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
        
        # Complex emotions formed by combining primary emotions
            'love': {
                'components': ['joy', 'trust'],
                'mood': 'Warm, connected, and devoted',
                'thoughts': 'This person/thing is wonderful and safe',
                'responses': 'Protective, nurturing, wants to be close',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'submission': {
                'components': ['trust', 'fear'],
                'mood': 'Deferential and compliant',
                'thoughts': 'I should follow their lead',
                'responses': 'Obeys, seeks approval, avoids conflict',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'awe': {
                'components': ['fear', 'surprise'],
                'mood': 'Humbled and overwhelmed',
                'thoughts': 'This is beyond my understanding',
                'responses': 'Reverent behavior, may feel small or insignificant',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'disapproval': {
                'components': ['surprise', 'sadness'],
                'mood': 'Disappointed and let down',
                'thoughts': 'This isn\'t what I expected or hoped for',
                'responses': 'Expresses dissatisfaction, may withdraw support',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'remorse': {
                'components': ['sadness', 'disgust'],
                'mood': 'Regretful and self-reproaching',
                'thoughts': 'I did something wrong and feel bad about it',
                'responses': 'Apologizes, seeks to make amends, self-punishing',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'contempt': {
                'components': ['disgust', 'anger'],
                'mood': 'Superior and disdainful',
                'thoughts': 'This is beneath me and doesn\'t deserve respect',
                'responses': 'Dismissive, condescending, may ridicule',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'aggressiveness': {
                'components': ['anger', 'anticipation'],
                'mood': 'Hostile and ready for conflict',
                'thoughts': 'I need to attack before they do',
                'responses': 'Threatening behavior, seeks confrontation',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            },
            'optimism': {
                'components': ['anticipation', 'joy'],
                'mood': 'Hopeful and positive about the future',
                'thoughts': 'Good things are coming',
                'responses': 'Plans enthusiastically, encourages others',
                'embedding': np.random.normal(0, 1, 128)  # Placeholder embedding
            }
        }

        self.ee_initialize_emotion_embeddings()

        self.default_emotion = "joy"  # Default emotion if none found

    def ee_initialize_emotion_embeddings(self):
        """
        Initialize embeddings for all emotions in the database
        
        Args:
            force_reinitialize (bool): If True, will re-embed emotions even if already done
        """
       
        for emotion in self.ee_emotion_database.keys():
            # Generate embedding using the embedding model
            em = self.ee_emotion_database[emotion]
            embedding = self.embedding_model.embed_text_string(em['mood'] + "," + em['thoughts'] + "," + em['responses'])
            self.ee_emotion_database[emotion]['embedding'] = embedding
            print(f"Initialized embedding for emotion: {emotion} (shape: {embedding.shape})")
    
    def get_emotional_reaction(self,text: str) -> Tuple[str, np.ndarray]:
        """
        Generate an emotional reaction to an input text.
        
        Args:
            text: Input text to analyze emotion ebedding similarity
            
        Returns:
            Tuple[str, Dict]: (emotion, emotion data)
        """
        #embed the input text with the same model used for emotions
        input_embedding = self.embedding_model.embed_text_string(text)

        # Check if input_embedding is valid
        if input_embedding is None:
            print("Error: Input text embedding is None")
            return self.default_emotion, self.ee_emotion_database[self.default_emotion]
        if input_embedding.shape[0] != 128:
            print(f"Error: Input embedding shape is {input_embedding.shape}, expected (128,)")
            return self.default_emotion, self.ee_emotion_database[self.default_emotion]
        # Ensure input_embedding is a 1D array
        if input_embedding.ndim != 1:
            print(f"Error: Input embedding is not 1D, shape is {input_embedding.shape}")
            return self.default_emotion, self.ee_emotion_database[self.default_emotion]
        
        # Find the closest emotion embedding
        best_emotion = self.default_emotion
        best_similarity = -1.0
        for emotion, data in self.ee_emotion_database.items():
            emotion_embedding = data['embedding']
            if emotion_embedding is None or emotion_embedding.shape != (128,):
                continue
            
            # Calculate cosine similarity between input and emotion embeddings
            similarity = np.dot(input_embedding, emotion_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(emotion_embedding)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_emotion = emotion
        
        print(f"Best matching emotion: {best_emotion} (similarity: {best_similarity:.3f})")
        return best_emotion, self.ee_emotion_database[best_emotion]
        
        
        

@dataclass
class MemoryPage:
    """
    Structured data class representing a memory page with multimodal embeddings
    and temporal/associative links to other memories.
    """
    # Core content
    text: Optional[str] = None
    text_embedding: Optional[np.ndarray] = None
    
    # Multimodal placeholders
    image_file: Optional[str] = None
    image_embedding: Optional[np.ndarray] = None
    audio_file: Optional[str] = None
    audio_embedding: Optional[np.ndarray] = None
    
    # Emotional and temporal context
    emotion: Optional[str] = None
    emotion_embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    
    # Associative links
    related_memory_ids: List[str] = field(default_factory=list)
    
    # Metadata
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    creation_datetime: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Ensure memory_id is always a string"""
        if not isinstance(self.memory_id, str):
            self.memory_id = str(self.memory_id)

@dataclass
class MemoryConcept:
    """Hierarchical memory cluster for storing thematically related memories"""
    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    theme: str = ""  # LLM-generated theme description
    summary: str = ""  # LLM-generated summary of all memories in cluster
    summary_embedding: Optional[np.ndarray] = None
    
    # Cluster metadata
    memory_ids: List[str] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    access_count: int = 0
    importance_score: float = 0.0
    
    # Temporal span - single averaged timestamp
    representative_time: float = 0.0  # Average time of all memories in cluster
    earliest_memory: float = 0.0
    latest_memory: float = 0.0
    temporal_span_days: float = 0.0
    
    # Semantic analysis
    key_entities: List[str] = field(default_factory=list)  # People, places, things
    key_nouns: List[str] = field(default_factory=list)
    common_themes: List[str] = field(default_factory=list)
    
    # Consolidation metadata
    consolidation_strength: float = 0.0  # How well-formed this concept is
    sub_concepts: List[str] = field(default_factory=list)  # Child concept IDs
    parent_concept: Optional[str] = None  # Parent concept ID

    # Multimodal embedding lists
    text_embeddings: List[np.ndarray] = field(default_factory=list)
    image_embeddings: List[np.ndarray] = field(default_factory=list)
    audio_embeddings: List[np.ndarray] = field(default_factory=list)
    
    # Computed average embeddings for quick similarity checks
    text_embedding: Optional[np.ndarray] = None
    image_embedding: Optional[np.ndarray] = None
    audio_embedding: Optional[np.ndarray] = None



class Hippocampus:
    """
    Memory consolidation system that creates conceptual clusters from individual memories.
    Inherits from PrefrontalCortex to access the memory store and embeddings.
    """
    
    def __init__(self, audio_embedder: Type[pseudo_vggish_embed],img_embedder: Type[pseudo_beit_embed], txt_embedder: Type[MxBaiEmbedder] = MxBaiEmbedder, memory_store_file: str = "memory_store.pkl",concept_store_file: str = "concept_store.pkl"):
        """
        Initialize Hippocampus with access to PrefrontalCortex
        
        Args:            
            audio_embedder: Class for audio embedding (e.g., pseudo_vggish_embed)
            img_embedder: Class for image embedding (e.g., pseudo_beit_embed)
            txt_embedder: Class for text embedding (e.g., MxBaiEmbedder)
            memory_store_file: Path to pickle file for persistent memory storage
            concept_store_file: Path to pickle file for persistent concept storage
        """

        self.txt_embedder = txt_embedder
        self.audio_embedder = audio_embedder
        self.img_embedder = img_embedder
        self.concept_store_file = concept_store_file
        
        # Load the text embedding model
        print("Loading text embedding model...")
        if not self.txt_embedder.load_model():
            print("Warning: Failed to load text embedding model. Text embeddings may not work properly.")
        else:
            print("Text embedding model loaded successfully.")

        self.memories = []  # List of MemoryPage objects
        self.memory_concepts = []  # List of MemoryConcept objects
        
        # Initialize spaCy for noun extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Clustering parameters
        self.semantic_similarity_threshold = 0.75  # For initial grouping
        self.temporal_window_hours = 24  # Memories within 24 hours can cluster
        self.min_cluster_size = 2
        self.max_cluster_size = 10
        self.entity_weight = 0.3  # How much to weight entity differences

        self._memory_save_file = memory_store_file
        self._concept_save_file = concept_store_file

        self._load_memory_store()
        
       
    def process_multimodal_input(self, 
                               text: Optional[str] = None, 
                               image_file: Optional[str] = None, 
                               audio_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multimodal input and manage memory, focus, and concept associations
        
        Args:
            text: Optional text input
            image_file: Optional image file path
            audio_file: Optional audio file path
            
        Returns:
            Dict containing:
            - 'memory_page': Created MemoryPage
            - 'similar_memories': List of similar MemoryPages
            - 'similar_concepts': List of similar MemoryConcepts
            - 'new_focus': Suggested focus (if any)
        """
        # Create embeddings for input
        text_embedding = self.embed_text_string(text) if text else None
        image_embedding = self.embed_image_file(image_file) if image_file else None
        audio_embedding = self.embed_audio_file(audio_file) if audio_file else None
        
        # Create a new memory page
        memory_page = MemoryPage(
            text=text,
            text_embedding=text_embedding,
            image_file=image_file,
            image_embedding=image_embedding,
            audio_file=audio_file,
            audio_embedding=audio_embedding,
            emotion=self.emotional_state,
            emotion_embedding=self.emotional_state_vector,
            timestamp=time.time()
        )
        
        # Search for similar memories
        similar_memories = self._find_similar_memories(memory_page)
        
        # Search for similar concepts
        similar_concepts = self._find_similar_concepts(memory_page)
        
        # Determine potential focus
        new_focus = self._determine_focus(memory_page, similar_memories, similar_concepts)
        
        # Update memory store and concepts
        self._update_memory_store(memory_page, similar_memories, similar_concepts)
        
        # Return processing results
        return {
            'memory_page': memory_page,
            'similar_memories': similar_memories,
            'similar_concepts': similar_concepts,
            'new_focus': new_focus
        }

    def _find_similar_memories(self, memory_page: MemoryPage, similarity_threshold: float = 0.7) -> List[MemoryPage]:
        """
        Find similar memories across multiple modalities
        
        Args:
            memory_page: Input memory page to compare
            similarity_threshold: Minimum similarity to consider
            
        Returns:
            List of similar MemoryPages
        """
        similar_memories = []
        
        for existing_memory in self.hippocampus.memories:
            similarities = []
            
            # Text similarity
            if (memory_page.text_embedding is not None and 
                existing_memory.text_embedding is not None):
                text_sim = cosine_similarity(
                    [memory_page.text_embedding], 
                    [existing_memory.text_embedding]
                )[0][0]
                similarities.append(text_sim)
            
            # Image similarity
            if (memory_page.image_embedding is not None and 
                existing_memory.image_embedding is not None):
                image_sim = cosine_similarity(
                    [memory_page.image_embedding], 
                    [existing_memory.image_embedding]
                )[0][0]
                similarities.append(image_sim)
            
            # Audio similarity
            if (memory_page.audio_embedding is not None and 
                existing_memory.audio_embedding is not None):
                audio_sim = cosine_similarity(
                    [memory_page.audio_embedding], 
                    [existing_memory.audio_embedding]
                )[0][0]
                similarities.append(audio_sim)
            
            # Compute overall similarity
            if similarities and np.mean(similarities) >= similarity_threshold:
                similar_memories.append(existing_memory)
        
        return similar_memories


    def search_similar_text_memories(self, text: str) -> List[MemoryPage]:
        """
        Search for similar memories based on text content.
        
        Args:
            text: Text content to search for
            
        Returns:
            List[MemoryPage]: List of matching MemoryPage objects
        """
        if not self.memories:
            print("No memories stored yet.")
            return []
        
        # Embed the input text
        text_embedding = self.txt_embedder.embed_text_string(text)
        
        results = []
        
        for memory in self.memories:
            if memory.text_embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = cosine_similarity([text_embedding], [memory.text_embedding])[0][0]
            
            if similarity >= self.semantic_similarity_threshold:
                results.append(memory)
        
        return results

    def search_similar_image_memories(self, image_file: str) -> List[MemoryPage]:
        """
        Search for similar memories based on image content.
        
        Args:
            image_file: Path to the image file to search for
            
        Returns:
            List[MemoryPage]: List of matching MemoryPage objects
        """
        if not self.memories:
            print("No memories stored yet.")
            return []
        
        # Embed the input image
        image_embedding = self.img_embedder.pseudo_beit_embed(image_file)
        
        results = []
        
        for memory in self.memories:
            if memory.image_embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = cosine_similarity([image_embedding], [memory.image_embedding])[0][0]
            
            if similarity >= self.semantic_similarity_threshold:
                results.append(memory)
        
        return results
    
    def search_similar_audio_memories(self, audio_file: str) -> List[MemoryPage]:
        """
        Search for similar memories based on audio content.
        
        Args:
            audio_file: Path to the audio file to search for
            
        Returns:
            List[MemoryPage]: List of matching MemoryPage objects
        """
        if not self.memories:
            print("No memories stored yet.")
            return []
        
        # Embed the input audio
        audio_embedding = self.audio_embedder.pseudo_vggish_embed(audio_file)
        
        results = []
        
        for memory in self.memories:
            if memory.audio_embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = cosine_similarity([audio_embedding], [memory.audio_embedding])[0][0]
            
            if similarity >= self.semantic_similarity_threshold:
                results.append(memory)
        
        return results

    def search_similar_memories(self,text: str, image_file: Optional[str] = None, audio_file: Optional[str] = None) -> List[MemoryPage]:
        """
        Search for similar memories based on text, image, or audio content.
        
        Args:
            text: Text content to search for
            image_file: Optional path to an image file
            audio_file: Optional path to an audio file
            
        Returns:
            List[MemoryPage]: List of matching MemoryPage objects
        """
        if not self.memories:
            print("No memories stored yet.")
            return []
        
        # Embed the input text, image, and audio
        text_embedding = self.txt_embedder.embed_text_string(text) if text else None
        image_embedding = self.img_embedder.pseudo_beit_embed(image_file) if image_file else None
        audio_embedding = self.audio_embedder.pseudo_vggish_embed(audio_file) if audio_file else None
        
        results = []
        
        for memory in self.memories:
            # Calculate similarity scores
            text_similarity = cosine_similarity([text_embedding], [memory.text_embedding])[0][0] if text_embedding is not None and memory.text_embedding is not None else 0.0
            image_similarity = cosine_similarity([image_embedding], [memory.image_embedding])[0][0] if image_embedding is not None and memory.image_embedding is not None else 0.0
            audio_similarity = cosine_similarity([audio_embedding], [memory.audio_embedding])[0][0] if audio_embedding is not None and memory.audio_embedding is not None else 0.0
            
            # Combine scores with weights
            combined_score = (text_similarity + image_similarity + audio_similarity) / 3
            
            if combined_score >= self.semantic_similarity_threshold:
                results.append(memory)
        
        return results

    def create_memory_concepts(self):
        """
        Create conceptual clusters from individual memories.
        
        This method groups memories into concepts based on semantic similarity,
        temporal proximity, and other heuristics.
        """
        if not self.memories:
            print("No memories to cluster.")
            return
        
        print(f"Processing {len(self.memories)} memories for concept formation...")
        
        # Load existing concepts
        self._load_concept_store()
        
        # Process each memory to find matching concepts or create new ones
        for memory in self.memories:
            if memory.text_embedding is None:
                continue  # Skip memories without text embeddings
            
            # Extract key entities and nouns from the memory
            entities, nouns = self._extract_entities_and_nouns(memory.text)
            
            # Find existing concepts that might match this memory
            matching_concept = self._find_matching_concept(memory, entities, nouns)
            
            if matching_concept:
                # Update existing concept with new information
                self._update_concept_with_memory(matching_concept, memory, entities, nouns)
                print(f"Updated concept '{matching_concept.theme}' with memory from {memory.creation_datetime}")
            else:
                # Check if this memory should start a new concept by finding similar memories
                similar_memories = self._find_similar_memories_for_concept(memory)
                
                if len(similar_memories) >= self.min_cluster_size:
                    # Create new concept from similar memories
                    new_concept = self._create_new_concept(similar_memories, entities, nouns)
                    self.memory_concepts.append(new_concept)
                    print(f"Created new concept '{new_concept.theme}' with {len(similar_memories)} memories")
        
        # Post-process concepts: merge very similar ones and update hierarchies
        self._merge_similar_concepts()
        
        # Update concept importance scores
        self._update_concept_importance()
        
        # Save updated concepts
        self._save_concept_store()
        
        print(f"Concept formation complete. Total concepts: {len(self.memory_concepts)}")
    
    def _extract_entities_and_nouns(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract named entities and nouns from text using spaCy.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple[List[str], List[str]]: (entities, nouns)
        """
        entities = []
        nouns = []
        
        if not self.nlp or not text:
            return entities, nouns
        
        try:
            doc = self.nlp(text)
            print(f"Extracting entities and nouns from text: {text[:50]}...")
            
            # Extract named entities (people, places, organizations, etc.)
            for ent in doc.ents:
                print(f"Found entity: {ent.text} ({ent.label_})")
                if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT"]:
                    entities.append(ent.text.lower())
            
            # Extract important nouns
            for token in doc:
                if (token.pos_ == "NOUN" and 
                    not token.is_stop and 
                    len(token.text) > 2 and
                    token.text.lower() not in entities):
                    nouns.append(token.text.lower())
            
        except Exception as e:
            print(f"Error extracting entities: {e}")
        
        return list(set(entities)), list(set(nouns))
    
    def _find_matching_concept(self, memory: MemoryPage, entities: List[str], nouns: List[str]) -> Optional[MemoryConcept]:
        """
        Find an existing concept that matches the given memory.
        
        Args:
            memory: Memory to find concept for
            entities: Extracted entities from memory
            nouns: Extracted nouns from memory
            
        Returns:
            Optional[MemoryConcept]: Matching concept if found
        """
        best_match = None
        best_score = 0.0
        
        for concept in self.memory_concepts:
            score = 0.0
            
            # Semantic similarity via embedding
            if concept.summary_embedding is not None and memory.text_embedding is not None:
                semantic_sim = cosine_similarity([memory.text_embedding], [concept.summary_embedding])[0][0]
                score += semantic_sim * 0.6  # 60% weight on semantic similarity
            
            # Entity overlap
            entity_overlap = len(set(entities) & set(concept.key_entities))
            if concept.key_entities:
                entity_sim = entity_overlap / len(concept.key_entities)
                score += entity_sim * 0.3  # 30% weight on entity similarity
            
            # Noun overlap
            noun_overlap = len(set(nouns) & set(concept.key_nouns))
            if concept.key_nouns:
                noun_sim = noun_overlap / len(concept.key_nouns)
                score += noun_sim * 0.1  # 10% weight on noun similarity
            
            # Prefer concepts that have been updated recently or have high importance
            recency_boost = max(0, 1 - (time.time() - concept.last_updated) / (7 * 24 * 3600))  # Decay over week
            score += recency_boost * 0.1
            
            if score > best_score and score >= 0.6:  # Threshold for matching
                best_score = score
                best_match = concept
        
        return best_match
    
    def _find_similar_memories_for_concept(self, target_memory: MemoryPage) -> List[MemoryPage]:
        """
        Find memories similar to the target memory for potential concept creation.
        
        Args:
            target_memory: Memory to find similar memories for
            
        Returns:
            List[MemoryPage]: List of similar memories including target
        """
        similar_memories = [target_memory]
        
        if target_memory.text_embedding is None:
            return similar_memories
        
        for memory in self.memories:
            if (memory.memory_id == target_memory.memory_id or 
                memory.text_embedding is None):
                continue
            
            # Calculate similarity
            similarity = cosine_similarity([target_memory.text_embedding], [memory.text_embedding])[0][0]
            
            if similarity >= self.semantic_similarity_threshold:
                similar_memories.append(memory)
                
                # Limit cluster size
                if len(similar_memories) >= self.max_cluster_size:
                    break
        
        return similar_memories
    
    def _create_new_concept(self, memories: List[MemoryPage], entities: List[str], nouns: List[str]) -> MemoryConcept:
        """
        Create a new MemoryConcept from a group of similar memories.
        
        Args:
            memories: List of memories to include in concept
            entities: Entities extracted from memories
            nouns: Nouns extracted from memories
            
        Returns:
            MemoryConcept: New concept
        """
        # Collect all entities and nouns from all memories
        all_entities = set(entities)
        all_nouns = set(nouns)
        all_texts = []
        
        for memory in memories:
            if memory.text:
                all_texts.append(memory.text)
                mem_entities, mem_nouns = self._extract_entities_and_nouns(memory.text)
                all_entities.update(mem_entities)
                all_nouns.update(mem_nouns)
        
        # Generate theme and summary
        theme = self._generate_concept_theme(list(all_entities), list(all_nouns))
        summary = self._generate_concept_summary(all_texts, theme)
        
        # Calculate temporal span
        timestamps = [memory.timestamp for memory in memories if hasattr(memory, 'timestamp')]
        if not timestamps:
            timestamps = [time.time()]
        
        earliest = min(timestamps)
        latest = max(timestamps)
        representative = sum(timestamps) / len(timestamps)
        
        # Create concept
        concept = MemoryConcept(
            theme=theme,
            summary=summary,
            summary_embedding=self.txt_embedder.embed_text_string(summary) if summary else None,
            memory_ids=[mem.memory_id for mem in memories],
            key_entities=list(all_entities)[:10],  # Limit to top 10
            key_nouns=list(all_nouns)[:15],  # Limit to top 15
            representative_time=representative,
            earliest_memory=earliest,
            latest_memory=latest,
            temporal_span_days=(latest - earliest) / (24 * 3600),
            consolidation_strength=min(1.0, len(memories) / self.max_cluster_size)
        )
        
        return concept
    
    def _update_concept_with_memory(self, concept: MemoryConcept, memory: MemoryPage, entities: List[str], nouns: List[str]):
        """
        Update an existing concept with information from a new memory.
        
        Args:
            concept: Concept to update
            memory: New memory to incorporate
            entities: Entities from new memory
            nouns: Nouns from new memory
        """
        # Add memory to concept
        if memory.memory_id not in concept.memory_ids:
            concept.memory_ids.append(memory.memory_id)
        
        # Update entities and nouns
        concept.key_entities.extend([e for e in entities if e not in concept.key_entities])
        concept.key_nouns.extend([n for n in nouns if n not in concept.key_nouns])
        
        # Limit list sizes
        concept.key_entities = concept.key_entities[:10]
        concept.key_nouns = concept.key_nouns[:15]
        
        # Update temporal information
        memory_time = getattr(memory, 'timestamp', time.time())
        concept.latest_memory = max(concept.latest_memory, memory_time)
        concept.earliest_memory = min(concept.earliest_memory, memory_time)
        concept.temporal_span_days = (concept.latest_memory - concept.earliest_memory) / (24 * 3600)
        
        # Recalculate representative time
        all_times = [concept.representative_time] + [memory_time]
        concept.representative_time = sum(all_times) / len(all_times)
        
        # Update summary if significant new information
        if memory.text and len(concept.memory_ids) % 3 == 0:  # Update every 3 new memories
            all_texts = [memory.text]  # Include new memory text
            updated_summary = self._generate_concept_summary(all_texts, concept.theme)
            if updated_summary:
                concept.summary = updated_summary
                concept.summary_embedding = self.txt_embedder.embed_text_string(updated_summary)
        
        # Update metadata
        concept.last_updated = time.time()
        concept.consolidation_strength = min(1.0, len(concept.memory_ids) / self.max_cluster_size)
    
    def _generate_concept_theme(self, entities: List[str], nouns: List[str]) -> str:
        """
        Generate a theme description for a concept based on entities and nouns.
        
        Args:
            entities: List of entities
            nouns: List of nouns
            
        Returns:
            str: Generated theme
        """
        # Simple heuristic-based theme generation
        # In practice, you might use an LLM for this
        
        if entities:
            primary_entities = entities[:3]  # Top 3 entities
            if len(primary_entities) == 1:
                theme = f"Conversations about {primary_entities[0]}"
            else:
                theme = f"Discussions involving {', '.join(primary_entities)}"
        elif nouns:
            primary_nouns = nouns[:3]  # Top 3 nouns
            theme = f"Topics related to {', '.join(primary_nouns)}"
        else:
            theme = "General conversation"
        
        return theme
    
    def _generate_concept_summary(self, texts: List[str], theme: str) -> str:
        """
        Generate a summary for a concept based on memory texts.
        
        Args:
            texts: List of text content from memories
            theme: Theme of the concept
            
        Returns:
            str: Generated summary
        """
        if not texts:
            return f"A concept about {theme}"
        
        # Simple extractive summary - take first few sentences
        # In practice, you might use an LLM for better summarization
        combined_text = " ".join(texts)
        sentences = combined_text.split(". ")[:3]  # First 3 sentences
        summary = ". ".join(sentences)
        
        if len(summary) > 200:
            summary = summary[:200] + "..."
        
        return summary
    
    def _merge_similar_concepts(self):
        """
        Merge concepts that are very similar to avoid duplication.
        """
        concepts_to_remove = set()
        
        for i, concept1 in enumerate(self.memory_concepts):
            if concept1.cluster_id in concepts_to_remove:
                continue
                
            for j, concept2 in enumerate(self.memory_concepts[i+1:], i+1):
                if concept2.cluster_id in concepts_to_remove:
                    continue
                
                # Check similarity between concepts
                if (concept1.summary_embedding is not None and 
                    concept2.summary_embedding is not None):
                    
                    similarity = cosine_similarity([concept1.summary_embedding], 
                                                 [concept2.summary_embedding])[0][0]
                    
                    # High similarity threshold for merging
                    if similarity >= 0.9:
                        # Merge concept2 into concept1
                        concept1.memory_ids.extend(concept2.memory_ids)
                        concept1.memory_ids = list(set(concept1.memory_ids))  # Remove duplicates
                        
                        # Merge entities and nouns
                        concept1.key_entities.extend(concept2.key_entities)
                        concept1.key_nouns.extend(concept2.key_nouns)
                        concept1.key_entities = list(set(concept1.key_entities))[:10]
                        concept1.key_nouns = list(set(concept1.key_nouns))[:15]
                        
                        # Update temporal information
                        concept1.earliest_memory = min(concept1.earliest_memory, concept2.earliest_memory)
                        concept1.latest_memory = max(concept1.latest_memory, concept2.latest_memory)
                        concept1.temporal_span_days = (concept1.latest_memory - concept1.earliest_memory) / (24 * 3600)
                        
                        # Mark concept2 for removal
                        concepts_to_remove.add(concept2.cluster_id)
                        
                        print(f"Merged concept '{concept2.theme}' into '{concept1.theme}'")
        
        # Remove merged concepts
        self.memory_concepts = [c for c in self.memory_concepts if c.cluster_id not in concepts_to_remove]
    
    def _update_concept_importance(self):
        """
        Update importance scores for all concepts based on various factors.
        """
        for concept in self.memory_concepts:
            score = 0.0
            
            # More memories = higher importance
            score += len(concept.memory_ids) * 0.1
            
            # Recent activity = higher importance
            days_since_update = (time.time() - concept.last_updated) / (24 * 3600)
            recency_score = max(0, 1 - days_since_update / 30)  # Decay over 30 days
            score += recency_score * 0.3
            
            # Longer temporal span = potentially more important
            temporal_score = min(1.0, concept.temporal_span_days / 7)  # Normalize to week
            score += temporal_score * 0.2
            
            # Rich entity/noun content = higher importance
            content_richness = min(1.0, (len(concept.key_entities) + len(concept.key_nouns)) / 15)
            score += content_richness * 0.2
            
            # Consolidation strength
            score += concept.consolidation_strength * 0.2
            
            concept.importance_score = min(1.0, score)
    
    def _save_concept_store(self):
        """Save all concepts to the persistent store"""
        try:
            with open(self._concept_save_file, 'wb') as f:
                pickle.dump(self.memory_concepts, f)
            print(f"Saved {len(self.memory_concepts)} concepts to {self._concept_save_file}")
        except Exception as e:
            print(f"Error saving concept store: {str(e)}")
    
    def _load_concept_store(self):
        """Load concepts from the persistent store"""
        try:
            if os.path.exists(self._concept_save_file):
                with open(self._concept_save_file, 'rb') as f:
                    self.memory_concepts = pickle.load(f)
                print(f"Loaded {len(self.memory_concepts)} concepts from {self._concept_save_file}")
            else:
                print(f"No existing concept store found at {self._concept_save_file}")
                self.memory_concepts = []
        except Exception as e:
            print(f"Error loading concept store: {str(e)}")
            self.memory_concepts = []
    
    def get_concept_by_id(self, concept_id: str) -> Optional[MemoryConcept]:
        """
        Retrieve a concept by its ID.
        
        Args:
            concept_id: ID of the concept to retrieve
            
        Returns:
            Optional[MemoryConcept]: The concept if found
        """
        for concept in self.memory_concepts:
            if concept.cluster_id == concept_id:
                return concept
        return None 
       
    def save_memory_page(self, memory_page: MemoryPage):
        """
        Save a single memory page to the memory store
        
        Args:
            memory_page: MemoryPage object to save
        """
        if not isinstance(memory_page, MemoryPage):
            raise ValueError("memory_page must be an instance of MemoryPage")
        
        self.memories.append(memory_page)
        
        # Save to persistent storage
        self._save_memory_store()
    
    def _save_memory_store(self):
        """Save all memory pages to the persistent store"""
        try:
            with open(self._memory_save_file, 'wb') as f:
                pickle.dump(self.memories, f)
            print(f"Saved {len(self.memories)} memory pages to {self._memory_save_file}")
        except Exception as e:
            print(f"Error saving memory store: {str(e)}")

    def _load_memory_store(self):
        """Load memory pages from the persistent store"""
        try:
            if os.path.exists(self._memory_save_file):
                with open(self._memory_save_file, 'rb') as f:
                    self.memories = pickle.load(f)
                print(f"Loaded {len(self.memories)} memory pages from {self._memory_save_file}")
            else:
                print(f"No existing memory store found at {self._memory_save_file}")
        except Exception as e:
            print(f"Error loading memory store: {str(e)}")



class PrefrontalCortex(EmotionEngine):
    
    """
    Advanced memory management system for a conversational LLM that handles temporal association,
    memory condensation, and multimodal similarity search with decay functions, and emotional context.
    """
    
    def __init__(self,audio_embedder: Type[pseudo_vggish_embed],img_embedder: Type[pseudo_beit_embed], hippocampus: Type[Hippocampus], txt_embedder: Type[MxBaiEmbedder] = MxBaiEmbedder, memory_store_file: str = "memory_store.pkl"):
        """
        Initialize the PrefrontalCortex with an embedder instance
        
        Args:
            embedders: Instances for embedding operations
            memory_store_file: Path to pickle file for persistent memory storage
        """
        # Initialize embedders as instances
        self.txt_embedder = txt_embedder()
        self.audio_embedder = audio_embedder()
        self.img_embedder = img_embedder()
        
        # Load the text embedding model
        print("Loading text embedding model for PrefrontalCortex...")
        if not self.txt_embedder.load_model():
            print("Warning: Failed to load text embedding model for PrefrontalCortex. Text embeddings may not work properly.")
        else:
            print("Text embedding model loaded successfully for PrefrontalCortex.")
        
        super().__init__(embedding_model=txt_embedder)#EmotionEngine initialization
        
        self.memory_store_file = memory_store_file
        self.memory_pages: Dict[str, MemoryPage] = {}
        self.temporal_chain: List[str] = []  # Ordered list of recent memory IDs by time (short term memory list)
        self.hippocampus = hippocampus  # Instance of Hippocampus for conceptual memory management
        
        # Decay and ranking parameters
        self.temporal_boost_decay_rate = 0.1  #Rate at which recent memories lose relevance 'boost'
        self.emotion_boost_multiplier = 1.3  # Positive emotions get ranking boost
        self.foundational_memory_preference = 0.85  #Rate at which foundational memories are preferred, else pick a different memory from returned similarity search
        
        # Initialize conversation focus
        self.focus: Optional[List[MemoryConcept]] = None # Current conversation focus concept such as a person, place, or event. can focus on multiple things suchs as a person in a place where the person and place are concepts 
        self.gaze: Tuple[int, int] = (500, 500)  # Current gaze coordinates (x, y) for visual focus, default to center of screen
        self.boot_timestamp = self.clock()  # Current timestamp in human-readable format

        self.ee_initialize_emotion_embeddings() #inherited from EmotionEngine
        self.emotional_state = self.default_emotion #set initial emotional state to default
        self.emotional_state_data = self.ee_emotion_database[self.emotional_state]  # Get initial emotion data
        self.emotional_state_vector = self.emotional_state_data['embedding']  # Set initial emotional state vector, short cut since this info is also in the _state_data dict 

        
    def clock(self):
        """Cross-platform datetime formatting"""
        now = datetime.datetime.now()
        
        # Format using standard codes and manually remove leading zeros
        month = now.strftime("%B")
        day = str(now.day)  # This removes leading zero automatically
        year = now.strftime("%Y")
        hour = str(now.hour if now.hour <= 12 else now.hour - 12)
        if hour == "0":
            hour = "12"
        minute = now.strftime("%M")
        ampm = "am" if now.hour < 12 else "pm"
        
        return f"{month} {day}, {year} at {hour}:{minute}{ampm}"
    
    
    def embed_text_string(self, text: str) -> np.ndarray:
        """
        Embed a text string using the configured text embedder
        
        Args:
            text: Input text to embed
            
        Returns:
            np.ndarray: Text embedding vector
        """
        if not self.txt_embedder:
            raise ValueError("Text embedder not initialized")
        
        return self.txt_embedder.embed_text_string(text)
    
    def embed_audio_file(self, audio_file: str) -> np.ndarray:
        """
        Embed an audio file using the configured audio embedder
        
        Args:
            audio_file: Path to the audio file to embed
            
        Returns:
            np.ndarray: Audio embedding vector
        """
        if not self.audio_embedder:
            raise ValueError("Audio embedder not initialized")
        
        return self.audio_embedder.pseudo_vggish_embed(audio_file)
    
    def embed_image_file(self, image_file: str) -> np.ndarray:
        """
        Embed an image file using the configured image embedder
        
        Args:
            image_file: Path to the image file to embed
            
        Returns:
            np.ndarray: Image embedding vector
        """
        if not self.img_embedder:
            raise ValueError("Image embedder not initialized")
        
        return self.img_embedder.pseudo_beit_embed(image_file)
    
    def send_data_to_hippocampus(self, text: Optional[str] = None,
                                  image_file: Optional[str] = None,
                                  audio_file: Optional[str] = None,
                                  emotion: Optional[str] = None) -> str:
        """        Send data to the Hippocampus for memory consolidation
        Args:
            text: Text content to store
            image_file: Path to an image file
            audio_file: Path to an audio file
            emotion: Emotional context of the memory
        Returns:
            str: Memory ID of the created memory page
        """
        if not (text or image_file or audio_file):
            raise ValueError("At least one content type (text, image, audio) must be provided")
        # Create a new MemoryPage
        memory_page = MemoryPage(
            text=text,
            text_embedding=self.embed_text_string(text) if text else None,
            image_embedding= self.embed_image_file(image_file) if image_file else None,
            audio_embedding= self.embed_audio_file(audio_file) if audio_file else None,
            image_file=image_file,
            audio_file=audio_file,
            emotion_embedding=self.emotional_state_vector if self.emotional_state_vector is not None else None,
            emotion=self.emotional_state if self.emotional_state is not None else emotion,
            timestamp=self.clock()
        )
        #send to hippocampus for consolidation
        memory_id = self.hippocampus.save_memory_page(memory_page)

        return memory_id
    
    def generate_emotional_response(self, text: str, set_emotional_state: bool = True) -> str:
        """
        Analyze text to determine emotional context and update current emotional state
        
        Args:
            text: Input text to analyze
            
        Returns:
            str: Detected emotion
        """
        emotion, emotion_data = self.get_emotional_reaction(text)
        if set_emotional_state:
            self.set_emotional_state(emotion, emotion_data)

    def set_emotional_state(self, emotion: str) -> None:
        """
        Set the current emotional state and update the emotional state vector
        
        Args:
            emotion: New emotional state (e.g., "happy", "sad")
            emotion_data: Optional additional data for the emotion {'mood': str, 'embedding': np.ndarray, 'thoughts': str, 'responses': str}
        """
        if emotion not in self.ee_emotion_database:
            raise ValueError(f"Unknown emotion: {emotion}")
        
        self.emotional_state = emotion
        self.emotional_state_data = self.ee_emotion_database[emotion]
        self.emotional_state_vector = self.emotional_state_data['embedding']

    def set_focus(self, concept_id: str) -> None:
        """
        Set the current conversation focus to a specific memory concept
        
        Args:
            concept_id: ID of the MemoryConcept to focus on
        """
        concept = self.hippocampus.get_concept_by_id(concept_id)
        if not concept:
            raise ValueError(f"Concept with ID {concept_id} not found")
        
        self.focus = [concept]
        print(f"Focus set to concept: {concept.theme} (ID: {concept.cluster_id})")
    
    def add_to_focus(self, concept_id: str) -> None:
        """
        Add a MemoryConcept to the current conversation focus
        
        Args:
            concept_id: ID of the MemoryConcept to add
        """
        concept = self.hippocampus.get_concept_by_id(concept_id)
        if not concept:
            raise ValueError(f"Concept with ID {concept_id} not found")
        
        if self.focus is None:
            self.focus = []
        
        # Avoid duplicates
        if concept not in self.focus:
            self.focus.append(concept)
            print(f"Added concept to focus: {concept.theme} (ID: {concept.cluster_id})")

    def remove_from_focus(self, concept_id: str) -> None:
        """
        Remove a MemoryConcept from the current conversation focus
        
        Args:
            concept_id: ID of the MemoryConcept to remove
        """
        if self.focus is None:
            print("No current focus to remove from")
            return
        
        concept = self.hippocampus.get_concept_by_id(concept_id)
        if not concept:
            raise ValueError(f"Concept with ID {concept_id} not found")
        
        if concept in self.focus:
            self.focus.remove(concept)
            print(f"Removed concept from focus: {concept.theme} (ID: {concept.cluster_id})")
        else:
            print(f"Concept {concept.theme} (ID: {concept.cluster_id}) not in current focus")

        # If focus is empty, reset it
        if not self.focus:
            self.focus = None
            print("Focus cleared")
    
    def set_gaze(self, x: int, y: int) -> None:
        """
        Set the current gaze coordinates for visual focus
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.gaze = (x, y)
        print(f"Gaze set to coordinates: ({x}, {y})")