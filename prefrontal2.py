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
        
        #lookup table for Plutchik's primary emotions to embeddings, these will be set with embedding_model
        self.ee_emotion_embedding_lookup  = {"joy": np.random.normal(0, 1, 128),"ecstasy": np.random.normal(0, 1, 128),
                                           "serenity": np.random.normal(0, 1, 128),"sadness": np.random.normal(0, 1, 128),
                                           "grief": np.random.normal(0, 1, 128),"pensiveness": np.random.normal(0, 1, 128),
                                           "trust": np.random.normal(0, 1, 128),"admiration": np.random.normal(0, 1, 128),
                                           "acceptance": np.random.normal(0, 1, 128),"disgust": np.random.normal(0, 1, 128),
                                             "loathing": np.random.normal(0, 1, 128),"boredom": np.random.normal(0, 1, 128),
                                             "fear": np.random.normal(0, 1, 128),"terror": np.random.normal(0, 1, 128),
                                             "apprehension": np.random.normal(0, 1, 128),"anger": np.random.normal(0, 1, 128),
                                             "rage": np.random.normal(0, 1, 128),"annoyance": np.random.normal(0, 1, 128),
                                             "surprise": np.random.normal(0, 1, 128),"amazement": np.random.normal(0, 1, 128),
                                                "distraction": np.random.normal(0, 1, 128),"anticipation": np.random.normal(0, 1, 128),
                                                "vigilance": np.random.normal(0, 1, 128),"interest": np.random.normal(0, 1, 128),
                                             "love": np.random.normal(0, 1, 128),"submission": np.random.normal(0, 1, 128),
                                                "awe": np.random.normal(0, 1, 128),"disapproval": np.random.normal(0, 1, 128),
                                                "remorse": np.random.normal(0, 1, 128),"contempt": np.random.normal(0, 1, 128),
                                                "aggressiveness": np.random.normal(0, 1, 128),"optimism": np.random.normal(0, 1, 128)}
                                           
        # Plutchik's primary emotions with their characteristics
        self.ee_emotion_database = {
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
            },
        
        # Complex emotions formed by combining primary emotions
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

        self.ee_initialize_emotion_embeddings()

        self.default_emotion = "joy"  # Default emotion if none found

    def ee_initialize_emotion_embeddings(self):
        """
        Initialize embeddings for all emotions in the database
        
        Args:
            force_reinitialize (bool): If True, will re-embed emotions even if already done
        """
       
        for emotion in self.ee_emotion_database.keys():
            if emotion in self.ee_emotion_embedding_lookup:
                # Generate embedding using the embedding model
                em = self.ee_emotion_database[emotion]
                embedding = self.embedding_model.embed_text_string(em['mood'] + "," + em['thoughts'] + "," + em['responses'])
                self.ee_emotion_embedding_lookup[emotion] = embedding
                print(f"Initialized embedding for emotion: {emotion} (shape: {embedding.shape})")
    
    def get_emotional_reaction(self,text: str) -> Tuple[str, np.ndarray]:
        """
        Generate an emotional reaction to an input text.
        
        Args:
            text: Input text to analyze emotion ebedding similarity
            
        Returns:
            Tuple[str, np.ndarray]: (emotion, embedding)
        """
        #ebedd the input text with the same model used for emotions
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
        closest_emotion = None
        closest_similarity = -1.0
        for emotion, embedding in self.ee_emotion_embedding_lookup.items():
            similarity = cosine_similarity([input_embedding], [embedding])[0][0]
            if similarity > closest_similarity:
                closest_similarity = similarity
                closest_emotion = emotion
        if closest_emotion:
            return closest_emotion, self.ee_emotion_database[closest_emotion]
        else:
            return self.default_emotion, self.ee_emotion_database[self.default_emotion]
        
        
        

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



class Hippocampus:
    """
    Memory consolidation system that creates conceptual clusters from individual memories.
    Inherits from PrefrontalCortex to access the memory store and embeddings.
    """
    
    def __init__(self, prefrontal_cortex, concept_store_file: str = "concept_store.pkl"):
        """
        Initialize Hippocampus with access to PrefrontalCortex
        
        Args:
            prefrontal_cortex: Instance of PrefrontalCortex class
            concept_store_file: Path to pickle file for persistent concept storage
        """
        self.prefrontal_cortex = prefrontal_cortex
        self.concept_store_file = concept_store_file
        self.memory_concepts: Dict[str, MemoryConcept] = {}
        # Set bidirectional reference
        prefrontal_cortex.set_hippocampus(self)
        
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
        
        # Load existing concepts
        self._load_concept_store()
    
    def auto_consolidate_all(self, min_age_hours: float = 1.0) -> Dict[str, int]:
        """
        Run all consolidation methods and return summary
        
        Args:
            min_age_hours: Minimum age for consolidation
            
        Returns:
            Dict with counts of different consolidation types
        """
        
        results = {
            'semantic_concepts': 0,
            'entity_concepts': 0, 
            'merged_memories': 0
        }
        
        # 1. Semantic consolidation (existing method)
        results['semantic_concepts'] = self.consolidate_memories(
            min_age_hours=min_age_hours,
            max_concepts_per_run=15
        )
        
        # 2. Entity-based consolidation  
        if hasattr(self.prefrontal_cortex, 'entity_clustering'):
            results['entity_concepts'] = self.prefrontal_cortex.entity_clustering.auto_consolidate_entity_clusters(
                max_clusters=10
            )
        
        # 3. Merge similar weak memories
        results['merged_memories'] = self.prefrontal_cortex.consolidate_similar_memories(
            max_merges=5
        )
        
        total_changes = sum(results.values())
        print(f"\nConsolidation Summary:")
        print(f"  Semantic concepts: {results['semantic_concepts']}")  
        print(f"  Entity concepts: {results['entity_concepts']}")
        print(f"  Merged memories: {results['merged_memories']}")
        print(f"  Total changes: {total_changes}")
        
        return results

    def _load_concept_store(self):
        """Load memory concepts from pickle file"""
        if os.path.exists(self.concept_store_file):
            try:
                with open(self.concept_store_file, 'rb') as f:
                    self.memory_concepts = pickle.load(f)
                print(f"Loaded {len(self.memory_concepts)} memory concepts from {self.concept_store_file}")
            except Exception as e:
                print(f"Error loading concept store: {str(e)}")
                self.memory_concepts = {}
        else:
            print(f"No existing concept store found at {self.concept_store_file}")
    
    def _save_concept_store(self):
        """Save memory concepts to pickle file"""
        try:
            with open(self.concept_store_file, 'wb') as f:
                pickle.dump(self.memory_concepts, f)
            print(f"Saved {len(self.memory_concepts)} memory concepts to {self.concept_store_file}")
        except Exception as e:
            print(f"Error saving concept store: {str(e)}")
    
    def extract_entities_and_nouns(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract entities, nouns, and key people/places from text using spaCy
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple[List[str], List[str], List[str]]: (entities, nouns, people_places)
        """
        if not self.nlp or not text:
            return [], [], []
        
        doc = self.nlp(text)
        
        # Extract named entities
        entities = [ent.text.lower() for ent in doc.ents]
        
        # Extract nouns (both common and proper)
        nouns = [token.lemma_.lower() for token in doc 
                if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
        
        # Extract people and places specifically
        people_places = [ent.text.lower() for ent in doc.ents 
                        if ent.label_ in ["PERSON", "GPE", "LOC", "ORG"]]
        
        return entities, nouns, people_places
    
    def calculate_entity_similarity(self, entities1: List[str], entities2: List[str]) -> float:
        """
        Calculate similarity between two sets of entities
        
        Args:
            entities1: First set of entities
            entities2: Second set of entities
            
        Returns:
            float: Similarity score (0-1)
        """
        if not entities1 and not entities2:
            return 1.0
        if not entities1 or not entities2:
            return 0.0
        
        set1 = set(entities1)
        set2 = set(entities2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_temporal_similarity(self, timestamp1: float, timestamp2: float) -> float:
        """
        Calculate temporal similarity between two timestamps
        
        Args:
            timestamp1: First timestamp
            timestamp2: Second timestamp
            
        Returns:
            float: Similarity score (0-1) based on temporal distance
        """
        time_diff_hours = abs(timestamp1 - timestamp2) / 3600
        
        if time_diff_hours <= self.temporal_window_hours:
            # Linear decay within the temporal window
            return 1.0 - (time_diff_hours / self.temporal_window_hours)
        else:
            # Exponential decay beyond the window
            return 0.1 * math.exp(-time_diff_hours / (self.temporal_window_hours * 2))
    
    def calculate_composite_similarity(self, memory1, memory2) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite similarity between two memories considering:
        - Semantic embedding similarity
        - Entity/noun overlap
        - Temporal proximity
        
        Args:
            memory1: First MemoryPage
            memory2: Second MemoryPage
            
        Returns:
            Tuple[float, Dict[str, float]]: (composite_score, breakdown)
        """
        breakdown = {}
        
        # 1. Semantic similarity from embeddings
        semantic_sim = 0.0
        if memory1.text_embedding is not None and memory2.text_embedding is not None:
            semantic_sim = np.dot(memory1.text_embedding, memory2.text_embedding)
        breakdown['semantic'] = semantic_sim
        
        # 2. Entity similarity
        entities1, nouns1, people_places1 = self.extract_entities_and_nouns(memory1.text or "")
        entities2, nouns2, people_places2 = self.extract_entities_and_nouns(memory2.text or "")
        
        entity_sim = self.calculate_entity_similarity(people_places1, people_places2)
        noun_sim = self.calculate_entity_similarity(nouns1, nouns2)
        breakdown['entities'] = entity_sim
        breakdown['nouns'] = noun_sim
        
        # 3. Temporal similarity
        temporal_sim = self.calculate_temporal_similarity(memory1.timestamp, memory2.timestamp)
        breakdown['temporal'] = temporal_sim
        
        # 4. Composite score with weights
        # If entities are very different (different people/places), reduce semantic weight
        entity_penalty = 1.0
        if entity_sim < 0.2 and semantic_sim > 0.7:
            entity_penalty = 0.5  # Reduce importance of semantic similarity
        
        composite_score = (
            semantic_sim * 0.4 * entity_penalty +
            entity_sim * 0.3 +
            noun_sim * 0.2 +
            temporal_sim * 0.1
        )
        
        breakdown['composite'] = composite_score
        breakdown['entity_penalty'] = entity_penalty
        
        return composite_score, breakdown
    
    def cluster_memories_by_similarity(self, memory_ids: List[str]) -> List[List[str]]:
        """
        Cluster memories using composite similarity (semantic + entity + temporal)
        
        Args:
            memory_ids: List of memory IDs to cluster
            
        Returns:
            List[List[str]]: List of clusters, each containing memory IDs
        """
        if len(memory_ids) < 2:
            return [memory_ids] if memory_ids else []
        
        # Get memory objects
        memories = [self.prefrontal_cortex.get_memory(mid) for mid in memory_ids]
        memories = [m for m in memories if m is not None]
        
        if len(memories) < 2:
            return [[m.memory_id for m in memories]] if memories else []
        
        # Calculate pairwise similarity matrix
        n = len(memories)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim_score, _ = self.calculate_composite_similarity(memories[i], memories[j])
                similarity_matrix[i][j] = sim_score
                similarity_matrix[j][i] = sim_score
        
        # Use DBSCAN clustering for flexible cluster sizes
        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        
        # DBSCAN parameters
        eps = 1 - self.semantic_similarity_threshold  # Distance threshold
        min_samples = self.min_cluster_size
        
        try:
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Group memories by cluster label
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label >= 0:  # -1 indicates noise/outlier
                    clusters[label].append(memories[i].memory_id)
                else:
                    # Create singleton cluster for outliers
                    clusters[f"outlier_{i}"] = [memories[i].memory_id]
            
            return list(clusters.values())
            
        except Exception as e:
            print(f"Error in clustering: {str(e)}")
            # Fallback: return individual memories as separate clusters
            return [[m.memory_id] for m in memories]
    
    def generate_concept_summary(self, memory_ids: List[str]) -> Tuple[str, str, List[str], List[str]]:
        """
        Generate theme and summary for a memory concept
        
        Args:
            memory_ids: List of memory IDs in the concept
            
        Returns:
            Tuple[str, str, List[str], List[str]]: (theme, summary, key_entities, key_nouns)
        """
        # Get memory texts
        memories = [self.prefrontal_cortex.get_memory(mid) for mid in memory_ids]
        memories = [m for m in memories if m and m.text]
        
        if not memories:
            return "Empty Concept", "No text content available", [], []
        
        # Combine all texts
        combined_text = " ".join([m.text for m in memories])
        
        # Extract common entities and nouns
        all_entities = []
        all_nouns = []
        for memory in memories:
            entities, nouns, people_places = self.extract_entities_and_nouns(memory.text)
            all_entities.extend(people_places)  # Focus on people/places for key entities
            all_nouns.extend(nouns)
        
        # Find most common entities and nouns
        entity_counts = Counter(all_entities)
        noun_counts = Counter(all_nouns)
        
        key_entities = [entity for entity, count in entity_counts.most_common(5)]
        key_nouns = [noun for noun, count in noun_counts.most_common(10)]
        
        # Generate simple theme based on most common elements
        if key_entities:
            theme = f"Activities involving {', '.join(key_entities[:3])}"
        elif key_nouns:
            theme = f"Experiences related to {', '.join(key_nouns[:3])}"
        else:
            theme = "General memories"
        
        # Generate simple summary
        if len(memories) == 1:
            summary = f"Single memory: {memories[0].text[:100]}..."
        else:
            summary = f"Collection of {len(memories)} related memories"
            if key_entities:
                summary += f" involving {', '.join(key_entities[:2])}"
            if key_nouns:
                summary += f" focused on {', '.join(key_nouns[:3])}"
        
        return theme, summary, key_entities, key_nouns
    
    def create_memory_concept(self, memory_ids: List[str]) -> MemoryConcept:
        """
        Create a MemoryConcept from a cluster of memory IDs
        
        Args:
            memory_ids: List of memory IDs to include in the concept
            
        Returns:
            MemoryConcept: The created concept
        """
        if not memory_ids:
            raise ValueError("Cannot create concept from empty memory list")
        
        # Get memory objects for temporal analysis
        memories = [self.prefrontal_cortex.get_memory(mid) for mid in memory_ids]
        memories = [m for m in memories if m]
        
        if not memories:
            raise ValueError("No valid memories found for concept creation")
        
        # Calculate temporal statistics
        timestamps = [m.timestamp for m in memories]
        earliest_time = min(timestamps)
        latest_time = max(timestamps)
        representative_time = sum(timestamps) / len(timestamps)  # Average timestamp
        temporal_span_days = (latest_time - earliest_time) / (24 * 3600)
        
        # Generate concept summary and theme
        theme, summary, key_entities, key_nouns = self.generate_concept_summary(memory_ids)
        
        # Create summary embedding from combined text
        summary_embedding = None
        if summary:
            try:
                summary_embedding = self.prefrontal_cortex.embedder.embed_text_string(summary)
            except Exception as e:
                print(f"Error creating summary embedding: {str(e)}")
        
        # Calculate importance score based on various factors
        importance_score = (
            len(memory_ids) * 0.3 +  # More memories = more important
            len(key_entities) * 0.2 +  # More entities = more important
            (1 / max(temporal_span_days + 1, 1)) * 0.3 +  # Recent clusters more important
            len(set(key_nouns)) * 0.2  # Diverse nouns = more important
        )
        
        # Create the concept
        concept = MemoryConcept(
            theme=theme,
            summary=summary,
            summary_embedding=summary_embedding,
            memory_ids=memory_ids.copy(),
            representative_time=representative_time,
            earliest_memory=earliest_time,
            latest_memory=latest_time,
            temporal_span_days=temporal_span_days,
            key_entities=key_entities,
            key_nouns=key_nouns,
            importance_score=importance_score,
            consolidation_strength=min(len(memory_ids) / self.max_cluster_size, 1.0)
        )
        
        return concept
    
    def consolidate_memories(self, min_age_hours: float = 1.0, max_concepts_per_run: int = 10) -> int:
        """
        Main consolidation function: cluster recent memories into concepts
        
        Args:
            min_age_hours: Minimum age of memories to consolidate (in hours)
            max_concepts_per_run: Maximum number of concepts to create in one run
            
        Returns:
            int: Number of concepts created
        """
        current_time = time.time()
        cutoff_time = current_time - (min_age_hours * 3600)
        
        # Find memories that are old enough to consolidate but not already in concepts
        already_conceptualized = set()
        for concept in self.memory_concepts.values():
            already_conceptualized.update(concept.memory_ids)
        
        candidate_memories = []
        for memory_id, memory in self.prefrontal_cortex.memory_pages.items():
            if (memory.timestamp < cutoff_time and 
                memory_id not in already_conceptualized and
                memory.text is not None):  # Only consolidate memories with text
                candidate_memories.append(memory_id)
        
        if len(candidate_memories) < self.min_cluster_size:
            print(f"Not enough candidate memories for consolidation: {len(candidate_memories)}")
            return 0
        
        print(f"Consolidating {len(candidate_memories)} candidate memories...")
        
        # Cluster the candidate memories
        clusters = self.cluster_memories_by_similarity(candidate_memories)
        
        # Filter clusters by size and create concepts
        concepts_created = 0
        for cluster in clusters:
            if (len(cluster) >= self.min_cluster_size and 
                len(cluster) <= self.max_cluster_size and
                concepts_created < max_concepts_per_run):
                
                try:
                    concept = self.create_memory_concept(cluster)
                    self.memory_concepts[concept.cluster_id] = concept
                    concepts_created += 1
                    print(f"Created concept: {concept.theme} ({len(cluster)} memories)")
                except Exception as e:
                    print(f"Error creating concept from cluster: {str(e)}")
        
        if concepts_created > 0:
            self._save_concept_store()
        
        print(f"Consolidation complete: {concepts_created} concepts created")
        return concepts_created
    
    def search_concepts(self, query_text: str, n: int = 5) -> List[Tuple[MemoryConcept, float]]:
        """
        Search memory concepts by text similarity
        
        Args:
            query_text: Text to search for
            n: Maximum number of results to return
            
        Returns:
            List[Tuple[MemoryConcept, float]]: List of (concept, similarity_score) tuples
        """
        if not query_text.strip() or not self.memory_concepts:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.prefrontal_cortex.embedder.embed_text_string(query_text)
            
            # Calculate similarity with concept summaries
            candidates = []
            for concept in self.memory_concepts.values():
                if concept.summary_embedding is not None:
                    similarity = np.dot(query_embedding, concept.summary_embedding)
                    # Boost by importance score
                    weighted_score = similarity * (1 + concept.importance_score * 0.1)
                    candidates.append((concept, weighted_score))
            
            # Sort by similarity and return top n
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:n]
            
        except Exception as e:
            print(f"Error in concept search: {str(e)}")
            return []
    
    def get_concept_by_id(self, concept_id: str) -> Optional[MemoryConcept]:
        """Get a concept by its ID and update access statistics"""
        if concept_id in self.memory_concepts:
            concept = self.memory_concepts[concept_id]
            concept.access_count += 1
            concept.last_updated = time.time()
            return concept
        return None
    
    def expand_concept(self, concept_id: str) -> List:
        """
        Expand a concept to retrieve its constituent memories
        
        Args:
            concept_id: ID of the concept to expand
            
        Returns:
            List[MemoryPage]: List of memory pages in the concept
        """
        concept = self.get_concept_by_id(concept_id)
        if not concept:
            return []
        
        memories = []
        for memory_id in concept.memory_ids:
            memory = self.prefrontal_cortex.get_memory(memory_id)
            if memory:
                memories.append(memory)
        
        # Sort by timestamp for chronological order
        memories.sort(key=lambda m: m.timestamp)
        return memories
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get statistics about memory consolidation"""
        total_concepts = len(self.memory_concepts)
        total_memories_in_concepts = sum(len(c.memory_ids) for c in self.memory_concepts.values())
        total_raw_memories = len(self.prefrontal_cortex.memory_pages)
        
        consolidation_ratio = total_memories_in_concepts / total_raw_memories if total_raw_memories > 0 else 0
        
        # Concept size distribution
        concept_sizes = [len(c.memory_ids) for c in self.memory_concepts.values()]
        avg_concept_size = sum(concept_sizes) / len(concept_sizes) if concept_sizes else 0
        
        return {
            'total_concepts': total_concepts,
            'total_memories_in_concepts': total_memories_in_concepts,
            'total_raw_memories': total_raw_memories,
            'consolidation_ratio': consolidation_ratio,
            'average_concept_size': avg_concept_size,
            'concept_size_distribution': Counter(concept_sizes)
        }
    
    def clear_all_concepts(self):
        """Clear all stored concepts"""
        self.memory_concepts.clear()
        self._save_concept_store()
        print("All memory concepts cleared")



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
        super().__init__(embedding_model=txt_embedder)#EmotionEngine initialization
        self.txt_embedder = txt_embedder
        self.audio_embedder = audio_embedder
        self.img_embedder = img_embedder
        self.memory_store_file = memory_store_file
        self.memory_pages: Dict[str, MemoryPage] = {}
        self.temporal_chain: List[str] = []  # Ordered list of recent memory IDs by time (short term memory list)
        self.hippocampus = hippocampus  # Instance of Hippocampus for conceptual memory management
        
        # Decay and ranking parameters
        self.temporal_boost_decay_rate = 0.1  #Rate at which recent memories lose relevance 'boost'
        self.emotion_boost_multiplier = 1.3  # Positive emotions get ranking boost
        self.foundational_memory_preference = 0.85  #Rate at which foundational memories are preferred, else pick a different memory from returned similarity search
        self.current_emotional_state_vector: Optional[np.ndarray] = None # Current emotional state vector for contextual memory retrieval
        self.current_emotion_state: Optional[str] = None  # Current emotional state as a string (e.g., "happy", "sad")
        
        # Initialize conversation focus
        self.focus =  Optional[MemoryConcept] = None # Current conversation focus concept such as a person, place, or event
        self.gaze = Tuple[int, int] = (0, 0)  # Current gaze coordinates (x, y) for visual focus
        self.boot_timestamp = self.clock()  # Current timestamp in human-readable format

        self.ee_initialize_emotion_embeddings() #inherited from EmotionEngine
        
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
            emotion_embedding=self.current_emotional_state_vector if self.current_emotional_state_vector is not None else None,
            emotion=self.current_emotion_state if self.current_emotion_state is not None else emotion,
            timestamp=self.clock()
        )
        #send to hippocampus for consolidation
        memory_id = self.hippocampus.create_memory_concept(memory_page)

        return memory_id
    
    def generate_emotional_response(self, text: str, set_emotional_state: bool = True) -> str:
        """
        Analyze text to determine emotional context and update current emotional state
        
        Args:
            text: Input text to analyze
            
        Returns:
            str: Detected emotion
        """
        # Placeholder for actual emotion detection logic
        # For demonstration, we'll use a simple keyword-based approach
        positive_keywords = ["happy", "joy", "love", "excited", "great", "fantastic"]
        negative_keywords = ["sad", "angry", "hate", "terrible", "bad", "upset"]
        
        text_lower = text.lower()
        if any(word in text_lower for word in positive_keywords):
            detected_emotion = "happy"
            self.current_emotional_state_vector = np.array([1.0, 0.0, 0.0])


    
    def clear_all_memories(self):
        """Clear all stored memories"""
        self.memory_pages.clear()
        self.temporal_chain.clear()
        self._save_memory_store()
        print("All memories cleared")
