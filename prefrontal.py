from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
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


class PrefrontalCortex:
    """
    Advanced memory management system that handles temporal association,
    memory condensation, and multimodal similarity search with decay functions.
    """
    
    def __init__(self, embedder, memory_store_file: str = "memory_store.pkl"):
        """
        Initialize the PrefrontalCortex with an embedder instance
        
        Args:
            embedder: Instance of MxBaiEmbedder for text embedding operations
            memory_store_file: Path to pickle file for persistent memory storage
        """
        self.embedder = embedder
        self.memory_store_file = memory_store_file
        self.memory_pages: Dict[str, MemoryPage] = {}
        self.temporal_chain: List[str] = []  # Ordered list of memory IDs by time
        
        # Decay and ranking parameters
        self.temporal_decay_rate = 0.85  # Newer memories favored 85% of time
        self.emotion_boost_multiplier = 1.3  # Positive emotions get ranking boost
        self.relation_boost_factor = 0.1  # Each related memory adds 10% to rank
        
        # Initialize entity clustering after hippocampus is available
        self.entity_clustering = None

    def set_hippocampus(self, hippocampus):
        """Set hippocampus reference and initialize entity clustering"""
        self.hippocampus = hippocampus
        self.entity_clustering = EntityBasedClustering(self, hippocampus)

        # Positive emotions based on Plutchik's wheel
        self.positive_emotions = {
            'ecstasy', 'joy', 'serenity',
            'admiration', 'trust', 'acceptance',
            'anticipation', 'vigilance', 'interest',
            'optimism', 'love',
            'submission', 'awe', 'satisfaction'
        }
        
        # Load existing memories
        self._load_memory_store()
    
    def _load_memory_store(self):
        """Load memory pages from pickle file"""
        if os.path.exists(self.memory_store_file):
            try:
                with open(self.memory_store_file, 'rb') as f:
                    data = pickle.load(f)
                    self.memory_pages = data.get('memory_pages', {})
                    self.temporal_chain = data.get('temporal_chain', [])
                print(f"Loaded {len(self.memory_pages)} memory pages from {self.memory_store_file}")
            except Exception as e:
                print(f"Error loading memory store: {str(e)}")
                self.memory_pages = {}
                self.temporal_chain = []
        else:
            print(f"No existing memory store found at {self.memory_store_file}")
    
    def _save_memory_store(self):
        """Save memory pages to pickle file"""
        try:
            data = {
                'memory_pages': self.memory_pages,
                'temporal_chain': self.temporal_chain
            }
            with open(self.memory_store_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {len(self.memory_pages)} memory pages to {self.memory_store_file}")
        except Exception as e:
            print(f"Error saving memory store: {str(e)}")
    
    def create_memory_page(self, 
                          text: Optional[str] = None,
                          text_embedding: Optional[np.ndarray] = None,
                          image_file: Optional[str] = None,
                          image_embedding: Optional[np.ndarray] = None,
                          audio_file: Optional[str] = None,
                          audio_embedding: Optional[np.ndarray] = None,
                          emotion: Optional[str] = None,
                          emotion_embedding: Optional[np.ndarray] = None,
                          auto_embed_text: bool = True,
                          auto_detect_emotion: bool = True) -> MemoryPage:
        """
        Create a new memory page with optional multimodal content
        
        Args:
            text: Text content (speech-to-text result or manual input)
            text_embedding: Precomputed text embedding (will compute if None and text provided)
            image_file: Path to image file
            image_embedding: Precomputed image embedding (placeholder)
            audio_file: Path to audio file
            audio_embedding: Precomputed audio embedding (placeholder)
            emotion: Emotion label
            emotion_embedding: Precomputed emotion embedding
            auto_embed_text: Whether to automatically embed text if not provided
            auto_detect_emotion: Whether to automatically detect emotion from text
            
        Returns:
            MemoryPage: The created memory page
        """
        # Auto-embed text if requested and available
        if text and text_embedding is None and auto_embed_text:
            try:
                text_embedding = self.embedder.embed_text_string(text)
            except Exception as e:
                print(f"Error embedding text: {str(e)}")
                text_embedding = None
        
        # Auto-detect emotion if requested and text available
        if text and emotion is None and auto_detect_emotion:
            try:
                if self.embedder.emotions_initialized:
                    emotion_name, similarity, _ = self.embedder.find_most_similar_emotion(text)
                    emotion = emotion_name
                    # Get emotion embedding from embedder's store
                    emotion_id = f"emotion_{emotion_name}"
                    if emotion_id in self.embedder.embeddings_store:
                        emotion_embedding = self.embedder.embeddings_store[emotion_id]
            except Exception as e:
                print(f"Error detecting emotion: {str(e)}")
        
        # Create the memory page
        memory_page = MemoryPage(
            text=text,
            text_embedding=text_embedding,
            image_file=image_file,
            image_embedding=image_embedding,
            audio_file=audio_file,
            audio_embedding=audio_embedding,
            emotion=emotion,
            emotion_embedding=emotion_embedding
        )
        
        return memory_page
    
    def _calculate_temporal_decay(self, memory_timestamp: float, current_time: float = None) -> float:
        """
        Calculate temporal decay factor for memory ranking
        
        Args:
            memory_timestamp: Timestamp of the memory
            current_time: Current timestamp (defaults to now)
            
        Returns:
            float: Decay factor (0-1, where 1 is no decay)
        """
        if current_time is None:
            current_time = time.time()
        
        # Calculate age in days
        age_seconds = current_time - memory_timestamp
        age_days = age_seconds / (24 * 3600)
        
        # Exponential decay: newer memories decay less
        # After 7 days, memory strength is reduced by ~15% (85% retention)
        decay_factor = math.exp(-age_days / 30)  # 30-day half-life
        return min(1.0, max(0.1, decay_factor))  # Clamp between 0.1 and 1.0
    
    def _calculate_memory_rank(self, memory_page: MemoryPage, similarity_score: float) -> Tuple[float, Dict]:
        """
        Calculate comprehensive ranking for a memory based on the correct sequence:
        1. Start with embedding similarity
        2. Apply temporal adjustment (85% favor newer)
        3. Apply popularity boost (more links = higher rank)
        
        Args:
            memory_page: The memory page to rank
            similarity_score: Base similarity score from embedding comparison
            
        Returns:
            Tuple[float, Dict]: (final_score, breakdown_dict)
        """
        breakdown = {}
        
        # Stage 1: Start with embedding similarity
        breakdown['embedding_similarity'] = similarity_score
        current_score = similarity_score
        
        # Stage 2: Apply temporal adjustment (85% favor newer memories)
        temporal_decay = self._calculate_temporal_decay(memory_page.timestamp)
        temporal_factor = 0.15 + 0.85 * temporal_decay  # 15% base + 85% decay-adjusted
        breakdown['temporal_factor'] = temporal_factor
        current_score *= temporal_factor
        
        # Stage 3: Apply popularity boost (more connections = more important)
        num_connections = len(memory_page.related_memory_ids)
        popularity_factor = 1 + (num_connections * self.relation_boost_factor)
        breakdown['popularity_factor'] = popularity_factor
        current_score *= popularity_factor
        
        # Optional: Boost positive emotions (can be applied at any stage)
        emotion_factor = 1.0
        if memory_page.emotion and memory_page.emotion.lower() in self.positive_emotions:
            emotion_factor = self.emotion_boost_multiplier
            current_score *= emotion_factor
        breakdown['emotion_factor'] = emotion_factor
        
        # Optional: Small boost for frequently accessed memories
        access_factor = 1 + (memory_page.access_count * 0.01)  # 1% per access
        access_factor = min(access_factor, 1.5)  # Cap at 50% boost
        breakdown['access_factor'] = access_factor
        current_score *= access_factor
        
        breakdown['final_score'] = current_score
        
        return current_score, breakdown
    
    def search_text(self, query_text: str, n: int = 5) -> List[Tuple[MemoryPage, float]]:
        """
        Search memory pages by text similarity
        
        Args:
            query_text: Text to search for
            n: Maximum number of results to return
            
        Returns:
            List[Tuple[MemoryPage, float]]: List of (memory_page, final_score) tuples
        """
        results = self.search_text_with_breakdown(query_text, n)
        return [(memory, final_score) for memory, final_score, _ in results]
    
    def search_text_with_breakdown(self, query_text: str, n: int = 5) -> List[Tuple[MemoryPage, float, Dict]]:
        """
        Search memory pages by text similarity with detailed ranking breakdown
        
        Args:
            query_text: Text to search for
            n: Maximum number of results to return
            
        Returns:
            List[Tuple[MemoryPage, float, Dict]]: List of (memory_page, final_score, breakdown) tuples
        """
        if not query_text.strip():
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedder.embed_text_string(query_text)
            
            # Find memories with text embeddings and calculate similarity
            candidates = []
            for memory_id, memory_page in self.memory_pages.items():
                if memory_page.text_embedding is not None:
                    # Calculate embedding similarity (cosine similarity)
                    similarity = np.dot(query_embedding, memory_page.text_embedding)
                    # Calculate comprehensive rank with breakdown
                    final_score, breakdown = self._calculate_memory_rank(memory_page, similarity)
                    candidates.append((memory_page, final_score, breakdown))
            
            # Sort by final score and return top n
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:n]
            
        except Exception as e:
            print(f"Error in text search: {str(e)}")
            return []
    
    def search_audio(self, query_audio_embedding: np.ndarray, n: int = 5) -> List[Tuple[MemoryPage, float]]:
        """
        Search memory pages by audio similarity (placeholder implementation)
        
        Args:
            query_audio_embedding: Audio embedding to search for
            n: Maximum number of results to return
            
        Returns:
            List[Tuple[MemoryPage, float]]: List of (memory_page, final_score) tuples
        """
        results = self.search_audio_with_breakdown(query_audio_embedding, n)
        return [(memory, final_score) for memory, final_score, _ in results]
    
    def search_audio_with_breakdown(self, query_audio_embedding: np.ndarray, n: int = 5) -> List[Tuple[MemoryPage, float, Dict]]:
        """
        Search memory pages by audio similarity with detailed ranking breakdown
        
        Args:
            query_audio_embedding: Audio embedding to search for
            n: Maximum number of results to return
            
        Returns:
            List[Tuple[MemoryPage, float, Dict]]: List of (memory_page, final_score, breakdown) tuples
        """
        candidates = []
        for memory_id, memory_page in self.memory_pages.items():
            if memory_page.audio_embedding is not None:
                # TODO: Implement actual audio similarity calculation
                # For now, using cosine similarity as placeholder
                similarity = np.dot(query_audio_embedding, memory_page.audio_embedding)
                final_score, breakdown = self._calculate_memory_rank(memory_page, similarity)
                candidates.append((memory_page, final_score, breakdown))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]
    
    def search_image(self, query_image_embedding: np.ndarray, n: int = 5) -> List[Tuple[MemoryPage, float]]:
        """
        Search memory pages by image similarity (placeholder implementation)
        
        Args:
            query_image_embedding: Image embedding to search for
            n: Maximum number of results to return
            
        Returns:
            List[Tuple[MemoryPage, float]]: List of (memory_page, final_score) tuples
        """
        results = self.search_image_with_breakdown(query_image_embedding, n)
        return [(memory, final_score) for memory, final_score, _ in results]
    
    def search_image_with_breakdown(self, query_image_embedding: np.ndarray, n: int = 5) -> List[Tuple[MemoryPage, float, Dict]]:
        """
        Search memory pages by image similarity with detailed ranking breakdown
        
        Args:
            query_image_embedding: Image embedding to search for
            n: Maximum number of results to return
            
        Returns:
            List[Tuple[MemoryPage, float, Dict]]: List of (memory_page, final_score, breakdown) tuples
        """
        candidates = []
        for memory_id, memory_page in self.memory_pages.items():
            if memory_page.image_embedding is not None:
                # TODO: Implement actual image similarity calculation
                # For now, using cosine similarity as placeholder
                similarity = np.dot(query_image_embedding, memory_page.image_embedding)
                final_score, breakdown = self._calculate_memory_rank(memory_page, similarity)
                candidates.append((memory_page, final_score, breakdown))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]
    
    def store_memory(self, memory_page: MemoryPage, link_to_similar: bool = True, 
                    similarity_threshold: float = 0.7,
                    identical_threshold: float = 0.95,
                    max_links: int = 5) -> str:
        """
        Store a memory page and create associative links
        
        Args:
            memory_page: The memory page to store
            link_to_similar: Whether to automatically link to similar memories
            similarity_threshold: Minimum similarity for automatic linking
            max_links: Maximum number of automatic links to create
            
        Returns:
            str: The memory ID of the stored memory
        """
        # Add temporal link to previous memory
        if self.temporal_chain:
            previous_memory_id = self.temporal_chain[-1]
            if previous_memory_id in self.memory_pages:
                # Link current memory to previous
                memory_page.related_memory_ids.append(previous_memory_id)
                # Link previous memory to current
                self.memory_pages[previous_memory_id].related_memory_ids.append(memory_page.memory_id)
        
        # Find and link similar memories
        if link_to_similar and memory_page.text is not None:
            similar_memories = self.search_text(memory_page.text, n=max_links * 2)
            
            links_added = 0
            for similar_memory, similarity_score in similar_memories:
                # Skip if it's the same memory or similarity too low
                if (similar_memory.memory_id == memory_page.memory_id or 
                    similarity_score < similarity_threshold or 
                    similarity_score > identical_threshold or  # Too similar = likely duplicate
                    links_added >= max_links):
                    continue
                
                # Create bidirectional links
                if similar_memory.memory_id not in memory_page.related_memory_ids:
                    memory_page.related_memory_ids.append(similar_memory.memory_id)
                    links_added += 1
                
                if memory_page.memory_id not in similar_memory.related_memory_ids:
                    similar_memory.related_memory_ids.append(memory_page.memory_id)
        
        # Store the memory
        self.memory_pages[memory_page.memory_id] = memory_page
        self.temporal_chain.append(memory_page.memory_id)
        
        # Save to disk
        self._save_memory_store()
        
        print(f"Stored memory {memory_page.memory_id} with {len(memory_page.related_memory_ids)} links")
        return memory_page.memory_id
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using the hippocampus NLP pipeline"""
        if hasattr(self, 'hippocampus') and self.hippocampus.nlp and text:
            doc = self.hippocampus.nlp(text)
            return [ent.text.lower() for ent in doc.ents if ent.label_ in ["PERSON", "GPE", "LOC", "ORG"]]
        return []

    def _find_entity_based_memories(self, entities: List[str]) -> List[Dict]:
        """Find memories based on entity matches"""
        if hasattr(self, 'entity_clustering'):
            return self.entity_clustering.find_entity_based_memories(entities)
        return []

    def _build_full_context(self, triggered_memories: List[Dict]) -> Dict[str, Any]:
        """Build comprehensive context from triggered memories"""
        
        context = {
            'total_triggered': len(triggered_memories),
            'by_type': defaultdict(int),
            'confidence_distribution': [],
            'temporal_spread': {},
            'key_entities': set(),
            'representative_memories': []
        }
        
        for tm in triggered_memories:
            trigger_type = tm['trigger_type']
            context['by_type'][trigger_type] += 1
            context['confidence_distribution'].append(tm['confidence'])
            
            if 'memory' in tm:
                memory = tm['memory']
                context['representative_memories'].append({
                    'text': memory.text[:100] + '...' if memory.text else '',
                    'timestamp': memory.timestamp,
                    'emotion': memory.emotion,
                    'confidence': tm['confidence']
                })
                
                # Extract entities from this memory
                entities = self._extract_entities(memory.text or '')
                context['key_entities'].update(entities)
        
        # Convert sets to lists for JSON serialization
        context['key_entities'] = list(context['key_entities'])
        context['by_type'] = dict(context['by_type'])
        
        return context

    def get_memory(self, memory_id: str) -> Optional[MemoryPage]:
        """
        Retrieve a memory page by ID and update access statistics
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Optional[MemoryPage]: The memory page if found, None otherwise
        """
        if memory_id in self.memory_pages:
            memory_page = self.memory_pages[memory_id]
            memory_page.access_count += 1
            memory_page.last_accessed = time.time()
            return memory_page
        return None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store"""
        total_memories = len(self.memory_pages)
        text_memories = sum(1 for m in self.memory_pages.values() if m.text is not None)
        audio_memories = sum(1 for m in self.memory_pages.values() if m.audio_file is not None)
        image_memories = sum(1 for m in self.memory_pages.values() if m.image_file is not None)
        
        total_links = sum(len(m.related_memory_ids) for m in self.memory_pages.values())
        avg_links = total_links / total_memories if total_memories > 0 else 0
        
        return {
            'total_memories': total_memories,
            'text_memories': text_memories,
            'audio_memories': audio_memories,
            'image_memories': image_memories,
            'total_associative_links': total_links,
            'average_links_per_memory': avg_links,
            'temporal_chain_length': len(self.temporal_chain)
        }
    
    def what_does_this_remind_me_of(self, 
                               text: Optional[str] = None,
                               text_embedding: Optional[np.ndarray] = None,
                               image_file: Optional[str] = None,
                               image_embedding: Optional[np.ndarray] = None,
                               audio_file: Optional[str] = None,
                               audio_embedding: Optional[np.ndarray] = None,
                               similarity_threshold: float = 0.6) -> Optional[MemoryPage]:
        """
        Core function: Take input, find what it reminds us of, save the input, return the association.
        Fixed to search BEFORE storing to prevent self-referencing.
        """
        
        # Step 1: Create embeddings for the input (but don't store yet)
        if text and text_embedding is None:
            text_embedding = self.embedder.embed_text_string(text)
        
        if image_file and image_embedding is None:
            # In real implementation, would generate image embedding
            image_embedding = pseudo_beit_embed(image_file)
                
        if audio_file and audio_embedding is None:
            # In real implementation, would generate audio embedding  
            audio_embedding = pseudo_vggish_embed(audio_file)
        
        # Step 2: Search for associations across ALL modalities BEFORE storing anything
        all_associations = []
        
        # Text associations
        if text:
            text_matches = self.search_text_with_breakdown(text, n=10)
            all_associations.extend([(memory, score, 'text', breakdown) 
                                for memory, score, breakdown in text_matches])
        
        # Audio associations
        if audio_embedding is not None:
            audio_matches = self.search_audio_with_breakdown(audio_embedding, n=5)
            all_associations.extend([(memory, score, 'audio', breakdown) 
                                for memory, score, breakdown in audio_matches])
        
        # Image associations  
        if image_embedding is not None:
            image_matches = self.search_image_with_breakdown(image_embedding, n=5)
            all_associations.extend([(memory, score, 'image', breakdown) 
                                for memory, score, breakdown in image_matches])
        
        # Step 3: Find the highest ranking association above threshold
        best_memory = None
        if all_associations:
            # Sort by score and get the best match across ALL modalities
            all_associations.sort(key=lambda x: x[1], reverse=True)
            best_memory_candidate, best_score, modality, breakdown = all_associations[0]
            
            if best_score >= similarity_threshold:
                # Update access statistics for the remembered memory
                best_memory_candidate.access_count += 1
                best_memory_candidate.last_accessed = time.time()
                best_memory = best_memory_candidate
                
                print(f"🧠 Association found via {modality}: score={best_score:.3f}")
        
        # Step 4: Create and store the new input memory (AFTER finding associations)
        new_memory = self.create_memory_page(
            text=text,
            text_embedding=text_embedding,
            image_file=image_file,
            image_embedding=image_embedding,
            audio_file=audio_file,
            audio_embedding=audio_embedding,
            auto_embed_text=False,  # We already created embeddings
            auto_detect_emotion=True
        )
        
        # Store with automatic linking (will link to similar memories, including the association we found)
        self.store_memory(new_memory, link_to_similar=True, similarity_threshold=0.5)
        
        # Step 5: Return the association (what this reminded us of) - could be from any modality
        return best_memory

    def identify_foundational_memories(self, min_references: int = 3, min_connections: int = 2) -> List[Tuple[MemoryPage, Dict]]:
        """
        Identify memories that have become foundational through repeated access and connections
        
        Args:
            min_references: Minimum access count to be considered foundational
            min_connections: Minimum number of connections to be considered foundational
            
        Returns:
            List[Tuple[MemoryPage, Dict]]: List of (memory, stats) for foundational memories
        """
        foundational = []
        current_time = time.time()
        
        for memory_id, memory in self.memory_pages.items():
            connections = len(memory.related_memory_ids)
            age_days = (current_time - memory.timestamp) / (24 * 3600)
            
            # Calculate foundational score based on multiple factors
            foundational_score = (
                memory.access_count * 0.4 +  # Access frequency
                connections * 0.3 +           # Network connectivity  
                (1 / max(age_days, 0.1)) * 0.2 +  # Recency bonus
                (1 if memory.emotion in self.positive_emotions else 0.8) * 0.1  # Emotion factor
            )
            
            if memory.access_count >= min_references and connections >= min_connections:
                stats = {
                    'access_count': memory.access_count,
                    'connections': connections,
                    'age_days': age_days,
                    'foundational_score': foundational_score
                }
                foundational.append((memory, stats))
        
        # Sort by foundational score
        foundational.sort(key=lambda x: x[1]['foundational_score'], reverse=True)
        return foundational
    
    def identify_weak_memories(self, max_references: int = 1, max_connections: int = 1, 
                              min_age_days: float = 1.0) -> List[Tuple[MemoryPage, Dict]]:
        """
        Identify memories that are weak candidates for consolidation or deletion
        
        Args:
            max_references: Maximum access count to be considered weak
            max_connections: Maximum connections to be considered weak
            min_age_days: Minimum age in days to be considered for removal
            
        Returns:
            List[Tuple[MemoryPage, Dict]]: List of (memory, stats) for weak memories
        """
        weak = []
        current_time = time.time()
        
        for memory_id, memory in self.memory_pages.items():
            connections = len(memory.related_memory_ids)
            age_days = (current_time - memory.timestamp) / (24 * 3600)
            
            if (memory.access_count <= max_references and 
                connections <= max_connections and 
                age_days >= min_age_days):
                
                stats = {
                    'access_count': memory.access_count,
                    'connections': connections,
                    'age_days': age_days,
                    'weakness_score': age_days / max(memory.access_count + connections + 1, 1)
                }
                weak.append((memory, stats))
        
        # Sort by weakness score (higher = weaker)
        weak.sort(key=lambda x: x[1]['weakness_score'], reverse=True)
        return weak
    
    def consolidate_similar_memories(self, similarity_threshold: float = 0.9, max_merges: int = 5) -> int:
        """
        Merge similar memories that have low activity to reduce redundancy
        
        Args:
            similarity_threshold: Minimum similarity to consider merging
            max_merges: Maximum number of merges to perform
            
        Returns:
            int: Number of memories that were merged/consolidated
        """
        merged_count = 0
        weak_memories = self.identify_weak_memories(max_references=2, max_connections=2)
        
        # Create list of weak memory embeddings for comparison
        weak_with_embeddings = [(memory, stats) for memory, stats in weak_memories 
                               if memory.text_embedding is not None]
        
        merged_ids = set()
        
        for i, (memory1, stats1) in enumerate(weak_with_embeddings):
            if memory1.memory_id in merged_ids or merged_count >= max_merges:
                continue
                
            for j, (memory2, stats2) in enumerate(weak_with_embeddings[i+1:], i+1):
                if memory2.memory_id in merged_ids:
                    continue
                
                # Calculate similarity
                similarity = np.dot(memory1.text_embedding, memory2.text_embedding)
                
                if similarity >= similarity_threshold:
                    # Merge memory2 into memory1 (keep the older one)
                    if memory1.timestamp <= memory2.timestamp:
                        primary, secondary = memory1, memory2
                    else:
                        primary, secondary = memory2, memory1
                    
                    # Combine text if both have it
                    if primary.text and secondary.text:
                        primary.text = f"{primary.text} [Merged: {secondary.text[:50]}...]"
                    
                    # Transfer connections
                    for related_id in secondary.related_memory_ids:
                        if related_id not in primary.related_memory_ids and related_id != primary.memory_id:
                            primary.related_memory_ids.append(related_id)
                            # Update bidirectional link
                            if related_id in self.memory_pages:
                                related_memory = self.memory_pages[related_id]
                                if secondary.memory_id in related_memory.related_memory_ids:
                                    related_memory.related_memory_ids.remove(secondary.memory_id)
                                if primary.memory_id not in related_memory.related_memory_ids:
                                    related_memory.related_memory_ids.append(primary.memory_id)
                    
                    # Combine access statistics
                    primary.access_count += secondary.access_count
                    
                    # Remove the secondary memory
                    del self.memory_pages[secondary.memory_id]
                    if secondary.memory_id in self.temporal_chain:
                        self.temporal_chain.remove(secondary.memory_id)
                    
                    merged_ids.add(secondary.memory_id)
                    merged_count += 1
                    break
        
        if merged_count > 0:
            self._save_memory_store()
        
        return merged_count
    
    def calculate_network_density(self) -> float:
        """Calculate the density of the memory network (how connected it is)"""
        if len(self.memory_pages) <= 1:
            return 0.0
        
        total_possible_connections = len(self.memory_pages) * (len(self.memory_pages) - 1)
        actual_connections = sum(len(memory.related_memory_ids) for memory in self.memory_pages.values())
        
        return actual_connections / total_possible_connections if total_possible_connections > 0 else 0.0
    
    def get_memory_strength_distribution(self) -> Dict[str, int]:
        """Get distribution of memory strengths in the network"""
        try:
            foundational_memories = self.identify_foundational_memories(min_references=3, min_connections=2)
            weak_memories = self.identify_weak_memories(max_references=1, max_connections=1)
            
            foundational = len(foundational_memories)
            weak = len(weak_memories)
            active = len(self.memory_pages) - foundational - weak
            
            return {
                'foundational': foundational,
                'active': max(0, active),
                'dormant': weak
            }
        except Exception as e:
            print(f"Error in get_memory_strength_distribution: {str(e)}")
            # Return a safe fallback
            return {
                'foundational': 0,
                'active': len(self.memory_pages),
                'dormant': 0
            }

    def process_new_experience(self, 
                           text: Optional[str] = None,
                           image_file: Optional[str] = None,
                           audio_file: Optional[str] = None,
                           return_full_context: bool = False) -> Dict[str, Any]:
        """
        Core memory processing flow: Input → Trigger → Gist → Store
        
        Args:
            text: New text input
            image_file: New image input
            audio_file: New audio input  
            return_full_context: Whether to return detailed context
            
        Returns:
            Dict containing triggered memories, gist, and storage confirmation
        """
        
        # Step 1: Create embeddings for new input (but don't store yet)
        input_embeddings = {}
        if text:
            input_embeddings['text'] = self.embedder.embed_text_string(text)
        if image_file:
            input_embeddings['image'] = pseudo_beit_embed(image_file)
        if audio_file:
            input_embeddings['audio'] = pseudo_vggish_embed(audio_file)
        
        # Step 2: Find triggered memories across all modalities
        triggered_memories = self._find_triggered_memories(
            text=text,
            embeddings=input_embeddings,
            threshold=0.6
        )
        
        # Step 3: Generate gist from triggered memories
        gist = self._generate_memory_gist(triggered_memories, text)
        
        # Step 4: Store new input as memory
        new_memory = self.create_memory_page(
            text=text,
            image_file=image_file, 
            audio_file=audio_file,
            auto_embed_text=True,
            auto_detect_emotion=True
        )
        
        memory_id = self.store_memory(new_memory, link_to_similar=True)
        
        # Return structured response
        result = {
            'triggered_memories': triggered_memories,
            'gist': gist,
            'new_memory_id': memory_id,
            'storage_confirmed': True
        }
        
        if return_full_context:
            result['full_context'] = self._build_full_context(triggered_memories)
            
        return result

    def _find_triggered_memories(self, text: str, embeddings: Dict, threshold: float) -> List[Dict]:
        """Find memories triggered by new input across all modalities"""
        
        all_triggered = []
        
        # Text-based triggers
        if text and 'text' in embeddings:
            text_matches = self.search_text_with_breakdown(text, n=5)
            for memory, score, breakdown in text_matches:
                if score >= threshold:
                    all_triggered.append({
                        'memory': memory,
                        'trigger_type': 'semantic',
                        'confidence': score,
                        'breakdown': breakdown
                    })
        
        # Concept-based triggers (from hippocampus)
        if hasattr(self, 'hippocampus') and text:
            concept_matches = self.hippocampus.search_concepts(text, n=3)
            for concept, score in concept_matches:
                if score >= threshold:
                    all_triggered.append({
                        'concept': concept,
                        'trigger_type': 'conceptual', 
                        'confidence': score,
                        'constituent_memories': len(concept.memory_ids)
                    })
        
        # Entity-based triggers
        entities = self._extract_entities(text) if text else []
        if entities:
            entity_matches = self._find_entity_based_memories(entities)
            all_triggered.extend(entity_matches)
        
        # Sort by confidence and return top triggers
        all_triggered.sort(key=lambda x: x['confidence'], reverse=True)
        return all_triggered[:10]  # Top 10 triggers

    def _generate_memory_gist(self, triggered_memories: List[Dict], current_input: str) -> Dict[str, str]:
        """Generate gist/summary of triggered memories"""
        
        if not triggered_memories:
            return {'summary': 'No related memories found', 'context': 'New experience'}
        
        # Separate memory types
        semantic_memories = [tm for tm in triggered_memories if tm['trigger_type'] == 'semantic']  
        conceptual_memories = [tm for tm in triggered_memories if tm['trigger_type'] == 'conceptual']
        entity_memories = [tm for tm in triggered_memories if tm['trigger_type'] == 'entity']
        
        gist = {
            'summary': '',
            'semantic_context': '',
            'conceptual_context': '', 
            'entity_context': '',
            'temporal_context': ''
        }
        
        # Semantic gist
        if semantic_memories:
            top_semantic = semantic_memories[0]['memory']
            gist['semantic_context'] = f"Similar to: {top_semantic.text[:100]}..."
            gist['summary'] = f"Reminds me of {len(semantic_memories)} similar experiences"
        
        # Conceptual gist  
        if conceptual_memories:
            top_concept = conceptual_memories[0]['concept']
            gist['conceptual_context'] = f"Related to concept: {top_concept.theme}"
            gist['summary'] += f", connects to '{top_concept.theme}' pattern"
            
        # Entity gist
        if entity_memories:
            gist['entity_context'] = f"Involves familiar entities: {', '.join([em['entity'] for em in entity_memories[:3]])}"
        
        # Temporal gist
        if semantic_memories:
            recent_memories = [sm for sm in semantic_memories if (time.time() - sm['memory'].timestamp) < 86400]  # 1 day
            if recent_memories:
                gist['temporal_context'] = f"Similar to {len(recent_memories)} recent experiences"
        
        return gist



    def clear_all_memories(self):
        """Clear all stored memories"""
        self.memory_pages.clear()
        self.temporal_chain.clear()
        self._save_memory_store()
        print("All memories cleared")


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
            similarity_threshold=0.85,
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


class EntityBasedClustering:
    """
    Advanced clustering based on proper nouns, people, places, and temporal patterns
    """
    
    def __init__(self, prefrontal_cortex, hippocampus):
        self.prefrontal_cortex = prefrontal_cortex
        self.hippocampus = hippocampus
        self.entity_memory_index = defaultdict(list)  # entity -> memory_ids
        self.temporal_clusters = defaultdict(list)    # time_bucket -> memory_ids
        
    def build_entity_index(self):
        """Build searchable index of entities to memories"""
        
        self.entity_memory_index.clear()
        
        for memory_id, memory in self.prefrontal_cortex.memory_pages.items():
            if not memory.text:
                continue
                
            # Extract entities using spaCy
            entities, nouns, people_places = self.hippocampus.extract_entities_and_nouns(memory.text)
            
            # Index by specific entity types
            for entity in people_places:  # People, places, organizations
                self.entity_memory_index[f"entity:{entity.lower()}"].append(memory_id)
                
            # Index by important nouns
            for noun in nouns[:5]:  # Top 5 nouns per memory
                self.entity_memory_index[f"noun:{noun.lower()}"].append(memory_id)
                
            # Index by time buckets (daily, weekly, monthly)
            timestamp = memory.timestamp
            date_key = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            week_key = datetime.fromtimestamp(timestamp).strftime('%Y-W%U')  
            month_key = datetime.fromtimestamp(timestamp).strftime('%Y-%m')
            
            self.temporal_clusters[f"day:{date_key}"].append(memory_id)
            self.temporal_clusters[f"week:{week_key}"].append(memory_id)
            self.temporal_clusters[f"month:{month_key}"].append(memory_id)
            
        print(f"Built entity index with {len(self.entity_memory_index)} entity keys")
        print(f"Built temporal index with {len(self.temporal_clusters)} time buckets")
    
    def find_entity_based_memories(self, entities: List[str], min_confidence: float = 0.7) -> List[Dict]:
        """Find memories that share specific entities"""
        
        entity_matches = []
        
        for entity in entities:
            entity_key = f"entity:{entity.lower()}"
            if entity_key in self.entity_memory_index:
                memory_ids = self.entity_memory_index[entity_key]
                
                for memory_id in memory_ids:
                    memory = self.prefrontal_cortex.get_memory(memory_id)
                    if memory:
                        entity_matches.append({
                            'memory': memory,
                            'entity': entity,
                            'trigger_type': 'entity',
                            'confidence': min_confidence + 0.1  # Slight boost for exact entity match
                        })
        
        return entity_matches
    
    def cluster_by_entity_and_time(self, time_window_days: int = 7) -> Dict[str, List[str]]:
        """
        Create clusters based on entity co-occurrence within time windows
        
        Args:
            time_window_days: Days within which memories can be clustered
            
        Returns:
            Dict mapping cluster_id to list of memory_ids
        """
        
        clusters = {}
        current_time = time.time()
        processed_memories = set()
        
        # Group by major entities first
        for entity_key, memory_ids in self.entity_memory_index.items():
            if not entity_key.startswith('entity:') or len(memory_ids) < 2:
                continue
                
            entity_name = entity_key.replace('entity:', '')
            
            # Sub-cluster by time within this entity group
            time_buckets = defaultdict(list)
            
            for memory_id in memory_ids:
                if memory_id in processed_memories:
                    continue
                    
                memory = self.prefrontal_cortex.get_memory(memory_id)
                if not memory:
                    continue
                    
                # Find time bucket (weekly grouping)
                week_key = datetime.fromtimestamp(memory.timestamp).strftime('%Y-W%U')
                time_buckets[week_key].append(memory_id)
            
            # Create clusters for entity+time combinations
            for week_key, week_memory_ids in time_buckets.items():
                if len(week_memory_ids) >= 2:  # At least 2 memories
                    cluster_id = f"{entity_name}_{week_key}"
                    clusters[cluster_id] = week_memory_ids
                    processed_memories.update(week_memory_ids)
        
        return clusters
    
    def auto_consolidate_entity_clusters(self, max_clusters: int = 20) -> int:
        """
        Automatically consolidate memories based on entity and temporal clustering
        
        Args:
            max_clusters: Maximum number of clusters to process
            
        Returns:
            int: Number of concepts created
        """
        
        self.build_entity_index()
        entity_clusters = self.cluster_by_entity_and_time()
        
        concepts_created = 0
        
        for cluster_id, memory_ids in list(entity_clusters.items())[:max_clusters]:
            
            # Skip if memories already in concepts
            already_consolidated = any(
                memory_id in concept.memory_ids 
                for concept in self.hippocampus.memory_concepts.values()
                for memory_id in memory_ids
            )
            
            if already_consolidated:
                continue
            
            try:
                # Create concept with entity-aware theming
                concept = self.create_entity_aware_concept(cluster_id, memory_ids)
                self.hippocampus.memory_concepts[concept.cluster_id] = concept
                concepts_created += 1
                
                print(f"Created entity-based concept: {concept.theme}")
                
            except Exception as e:
                print(f"Error creating entity cluster {cluster_id}: {str(e)}")
                
        if concepts_created > 0:
            self.hippocampus._save_concept_store()
            
        return concepts_created
    
    def create_entity_aware_concept(self, cluster_id: str, memory_ids: List[str]) -> 'MemoryConcept':
        """Create a concept with entity-aware theming"""
        
        # Extract entity and time from cluster_id
        parts = cluster_id.split('_')
        entity_name = parts[0] if parts else "Unknown"
        time_period = parts[1] if len(parts) > 1 else "Unknown"
        
        # Get memories for analysis
        memories = [self.prefrontal_cortex.get_memory(mid) for mid in memory_ids]
        memories = [m for m in memories if m and m.text]
        
        if not memories:
            raise ValueError("No valid memories for concept creation")
        
        # Generate entity-focused theme and summary
        theme = f"Experiences with {entity_name.title()}"
        if time_period != "Unknown":
            theme += f" during {time_period}"
        
        # Create summary focusing on the common entity
        summary = f"Collection of {len(memories)} memories involving {entity_name}"
        
        # Extract all entities for this concept
        all_entities = []
        all_nouns = []
        for memory in memories:
            entities, nouns, people_places = self.hippocampus.extract_entities_and_nouns(memory.text)
            all_entities.extend(people_places)
            all_nouns.extend(nouns)
        
        key_entities = [entity for entity, count in Counter(all_entities).most_common(5)]
        key_nouns = [noun for noun, count in Counter(all_nouns).most_common(10)]
        
        # Calculate temporal info
        timestamps = [m.timestamp for m in memories]
        earliest_time = min(timestamps)
        latest_time = max(timestamps)
        representative_time = sum(timestamps) / len(timestamps)
        temporal_span_days = (latest_time - earliest_time) / (24 * 3600)
        
        # Create summary embedding
        summary_embedding = None
        try:
            summary_embedding = self.prefrontal_cortex.embedder.embed_text_string(summary)
        except Exception as e:
            print(f"Error creating summary embedding: {str(e)}")
        
        # Calculate importance (entity-based concepts get bonus)
        importance_score = (
            len(memory_ids) * 0.4 +  # Size matters
            len(key_entities) * 0.3 +  # Entity diversity
            (1 / max(temporal_span_days + 1, 1)) * 0.2 +  # Temporal concentration
            1.5  # Entity-based bonus
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
            consolidation_strength=min(len(memory_ids) / 10, 1.0)
        )
        
        return concept
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about entity-based clustering"""
        
        entity_counts = {}
        for key, memory_ids in self.entity_memory_index.items():
            entity_type = key.split(':')[0]
            if entity_type not in entity_counts:
                entity_counts[entity_type] = 0
            entity_counts[entity_type] += len(memory_ids)
        
        temporal_counts = {}
        for key, memory_ids in self.temporal_clusters.items():
            time_type = key.split(':')[0]
            if time_type not in temporal_counts:
                temporal_counts[time_type] = 0
            temporal_counts[time_type] += 1
        
        return {
            'total_entities': len([k for k in self.entity_memory_index.keys() if k.startswith('entity:')]),
            'total_nouns': len([k for k in self.entity_memory_index.keys() if k.startswith('noun:')]),
            'entity_memory_distribution': entity_counts,
            'temporal_bucket_distribution': temporal_counts,
            'most_referenced_entities': sorted(
                [(k.replace('entity:', ''), len(v)) for k, v in self.entity_memory_index.items() if k.startswith('entity:')],
                key=lambda x: x[1], reverse=True
            )[:10]
        }


####################################

#!/usr/bin/env python3
"""
Brain-Like Memory System Demo
=============================

This demo showcases a complete memory system that mimics human memory:
1. Creates diverse memories with text, audio, and visual content
2. Uses real embeddings for semantic understanding
3. Consolidates memories into conceptual clusters (like the hippocampus)
4. Demonstrates memory recall and pruning

Usage:
    python memory_demo.py
"""


# Pseudo embedding functions 
#audio
def pseudo_vggish_embed(audio_file: str) -> np.ndarray:
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
def pseudo_beit_embed(image_file: str) -> np.ndarray:
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


class MemorySystemDemo:
    """
    Comprehensive demo of the brain-like memory system
    """
    
    def __init__(self):
        print("🧠 Initializing Brain-Like Memory System...")
        
        # Initialize the embedder
        self.embedder = MxBaiEmbedder()
        self.embedder.load_model()
        
        # Initialize the memory systems
        self.prefrontal_cortex = PrefrontalCortex(self.embedder)
        self.hippocampus = Hippocampus(self.prefrontal_cortex)
        
        # Memory generation templates
        self.simple_templates = [
            "Working on {task} - feeling {emotion}",
            "Just had {food} for {meal} - it was {quality}",
            "Saw {person} at {location} - we talked about {topic}",
            "Beautiful {weather} today - went for a {activity}",
            "Listening to {music} while {activity}",
            "Feeling {emotion} about {topic}",
            "Quick break - checking {app} and {activity}",
            "Meeting about {topic} - {outcome}",
            "{weather} outside - staying {location}",
            "Learning about {topic} - very {emotion}"
        ]
        
        self.complex_templates = [
            # Work scenarios
            "Had a productive meeting with the design team about the new product features",
            "Struggling with a difficult bug in the authentication system",
            "Presentation went really well - got positive feedback from the stakeholders", 
            "Feeling overwhelmed with the project deadline approaching next week",
            "Collaborated with Sarah on the API integration - good progress",
            "Coffee break conversation with Mike about improving our development process",
            
            # Personal life
            "Went for a morning run in Central Park - beautiful weather today",
            "Cooking dinner for friends tonight - trying a new pasta recipe",
            "Called mom to check how she's doing after her doctor appointment",
            "Finished reading 'Atomic Habits' - some really useful insights about behavior change",
            "Grocery shopping took forever - the store was packed on Sunday afternoon",
            "Watched an amazing documentary about ocean life with the kids",
            
            # Social interactions
            "Lunch with college friends - caught up on everyone's life changes",
            "Neighborhood barbecue was fun - met some new people from down the street",
            "Helped Emma move to her new apartment - she's excited about the change",
            "Game night at David's place - played some great board games",
            "Birthday party for my nephew - he loved the LEGO set I got him",
            
            # Learning and growth
            "Started learning Spanish using a new app - pronunciation is challenging",
            "Attended a webinar about machine learning applications in healthcare",
            "Practiced piano for an hour - finally getting the hang of that difficult piece",
            "Read an interesting article about sustainable urban planning",
            "Took an online course module about data visualization techniques",
            
            # Health and wellness
            "Yoga class was particularly relaxing today - felt great afterward", 
            "Dentist appointment went fine - no cavities this time",
            "Tried meditation for 15 minutes this morning - still finding it difficult to focus",
            "Made a healthy smoothie with spinach and berries for breakfast",
            "Evening walk around the neighborhood - good way to decompress after work",
            
            # Travel and experiences
            "Weekend trip to the mountains - stunning views from the hiking trail",
            "Visited the art museum downtown - loved the contemporary photography exhibition",
            "Flight was delayed by 2 hours but finally made it to the conference",
            "Exploring the local farmers market - bought some amazing fresh produce",
            "Road trip with friends to the coast - perfect weather for beach activities",
            
            # Technology and tools
            "Set up a new productivity system using Notion - seems promising so far",
            "Upgraded my laptop's RAM - noticeable improvement in performance",
            "Discovered a useful VS Code extension for better code formatting",
            "Smart home automation is working well - lights automatically adjust in the evening",
            "Backing up photos to cloud storage - realized I have thousands of old pictures",
            
            # Random daily life
            "Power went out for 3 hours - played board games by candlelight",
            "Package delivery was left with the wrong neighbor again",
            "Found a great new coffee shop on the way to work",
            "Car inspection is due next month - need to schedule an appointment",
            "Reorganized my bookshelf and found some books I forgot I owned"
        ]
        
        # Sample values for simple templates
        self.template_values = {
            'tasks': ["coding", "design", "writing", "planning", "debugging"],
            'emotions': ["focused", "tired", "excited", "stressed", "calm"],
            'foods': ["coffee", "sandwich", "salad", "pasta", "soup"],
            'meals': ["breakfast", "lunch", "dinner", "snack"],
            'quality': ["delicious", "okay", "terrible", "amazing", "bland"],
            'people': ["Sarah", "Mike", "Alex", "Emma", "David"],
            'locations': ["office", "cafe", "park", "gym", "home"],
            'topics': ["vacation", "work", "family", "technology", "movies"],
            'weather': ["sunny", "rainy", "cloudy", "snowy", "windy"],
            'activities': ["walk", "run", "read", "code", "rest"],
            'music': ["jazz", "rock", "classical", "pop", "ambient"],
            'apps': ["email", "news", "social media", "calendar", "messages"],
            'outcomes': ["productive", "confusing", "helpful", "long", "brief"]
        }
    
    def generate_memory_text(self) -> str:
        """Generate a random memory text"""
        if random.random() < 0.7:  # 70% chance of complex template
            return random.choice(self.complex_templates)
        else:  # 30% chance of simple template
            template = random.choice(self.simple_templates)
            # Fill in template with random values
            for key, values in self.template_values.items():
                if f"{{{key}}}" in template:
                    template = template.replace(f"{{{key}}}", random.choice(values))
            return template
    
    def create_diverse_memories(self, num_memories: int = 50) -> List[str]:
        """
        Create diverse memories with text, and some with audio/image
        
        Args:
            num_memories: Number of memories to create
            
        Returns:
            List[str]: List of created memory IDs
        """
        print(f"\n📝 Creating {num_memories} diverse memories...")
        
        memory_ids = []
        start_time = time.time() - (7 * 24 * 3600)  # Start 7 days ago
        
        for i in range(num_memories):
            # Generate memory text
            text = self.generate_memory_text()
            
            # Add some temporal variation (memories spread over past week)
            memory_time = start_time + (i * (7 * 24 * 3600) / num_memories)
            
            # Create memory page with text
            memory_page = self.prefrontal_cortex.create_memory_page(
                text=text,
                auto_embed_text=True,
                auto_detect_emotion=True
            )
            
            # Override timestamp to spread memories over time
            memory_page.timestamp = memory_time
            
            # 30% chance of adding audio file
            if random.random() < 0.3:
                audio_file = f"audio_recording_{i}.wav"
                memory_page.audio_file = audio_file
                memory_page.audio_embedding = pseudo_vggish_embed(audio_file)
            
            # 20% chance of adding image file
            if random.random() < 0.2:
                image_file = f"photo_{i}.jpg"
                memory_page.image_file = image_file
                memory_page.image_embedding = pseudo_beit_embed(image_file)
            
            # Store the memory
            memory_id = self.prefrontal_cortex.store_memory(
                memory_page, 
                link_to_similar=True,
                similarity_threshold=0.6
            )
            memory_ids.append(memory_id)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Created {i + 1}/{num_memories} memories...")
        
        print(f"✅ Created {len(memory_ids)} memories successfully!")
        return memory_ids
    
    def demonstrate_memory_recall(self, num_queries: int = 5):
        """Demonstrate memory recall functionality"""
        print(f"\n🔍 Demonstrating Memory Recall...")
        
        query_examples = [
            "working on coding projects",
            "meeting with friends",
            "feeling stressed about work",
            "going for a walk",
            "learning something new"
        ]
        
        for i, query in enumerate(query_examples[:num_queries]):
            print(f"\n--- Query {i+1}: '{query}' ---")
            
            # Search memories
            results = self.prefrontal_cortex.search_text_with_breakdown(query, n=3)
            
            if results:
                for j, (memory, score, breakdown) in enumerate(results):
                    print(f"  Result {j+1} (Score: {score:.3f}):")
                    print(f"    Text: {memory.text[:80]}...")
                    print(f"    Time: {datetime.fromtimestamp(memory.timestamp).strftime('%Y-%m-%d %H:%M')}")
                    print(f"    Emotion: {memory.emotion}")
                    print(f"    Breakdown: {breakdown}")
            else:
                print("  No matching memories found.")
    
    def demonstrate_association_recall(self):
        """Demonstrate the 'what does this remind me of' functionality"""
        print(f"\n🧠 Demonstrating Association Recall...")
        
        test_inputs = [
            "Had lunch with my colleague today",
            "Feeling excited about the weekend",
            "Working late on a difficult problem",
            "Beautiful sunny weather outside"
        ]
        
        for i, input_text in enumerate(test_inputs):
            print(f"\n--- Input {i+1}: '{input_text}' ---")
            
            # Use the hippocampus association function
            association = self.prefrontal_cortex.what_does_this_remind_me_of(
                text=input_text,
                similarity_threshold=0.6
            )
            
            if association:
                print(f"  🔗 This reminds me of:")
                print(f"    Text: {association.text[:80]}...")
                print(f"    Time: {datetime.fromtimestamp(association.timestamp).strftime('%Y-%m-%d %H:%M')}")
                print(f"    Emotion: {association.emotion}")
                print(f"    Access count: {association.access_count}")
            else:
                print("  No strong associations found.")
    
    def demonstrate_memory_consolidation(self):
        """Demonstrate hippocampus memory consolidation"""
        print(f"\n🏛️ Demonstrating Memory Consolidation (Hippocampus)...")
        
        # Show stats before consolidation
        stats_before = self.prefrontal_cortex.get_memory_stats()
        print(f"Before consolidation:")
        print(f"  Total memories: {stats_before['total_memories']}")
        print(f"  Text memories: {stats_before['text_memories']}")
        print(f"  Average links per memory: {stats_before['average_links_per_memory']:.2f}")
        
        # Perform consolidation
        print(f"\n🔄 Running memory consolidation...")
        concepts_created = self.hippocampus.consolidate_memories(
            min_age_hours=0.1,  # Very recent for demo
            max_concepts_per_run=15
        )
        
        # Show consolidation results
        consolidation_stats = self.hippocampus.get_consolidation_stats()
        print(f"\nConsolidation Results:")
        print(f"  Concepts created: {concepts_created}")
        print(f"  Total concepts: {consolidation_stats['total_concepts']}")
        print(f"  Memories in concepts: {consolidation_stats['total_memories_in_concepts']}")
        print(f"  Consolidation ratio: {consolidation_stats['consolidation_ratio']:.2%}")
        print(f"  Average concept size: {consolidation_stats['average_concept_size']:.1f}")
        
        # Show some example concepts
        print(f"\n📚 Example Memory Concepts:")
        for i, (concept_id, concept) in enumerate(list(self.hippocampus.memory_concepts.items())[:5]):
            print(f"\n  Concept {i+1}: {concept.theme}")
            print(f"    Summary: {concept.summary[:100]}...")
            print(f"    Memories: {len(concept.memory_ids)}")
            print(f"    Key entities: {', '.join(concept.key_entities[:3])}")
            print(f"    Time span: {concept.temporal_span_days:.1f} days")
            print(f"    Importance: {concept.importance_score:.2f}")
    
    def demonstrate_concept_search(self):
        """Demonstrate searching through consolidated concepts"""
        print(f"\n🔎 Demonstrating Concept Search...")
        
        search_queries = [
            "work meetings and collaboration",
            "exercise and outdoor activities",
            "learning and personal development"
        ]
        
        for query in search_queries:
            print(f"\n--- Searching concepts for: '{query}' ---")
            
            concept_results = self.hippocampus.search_concepts(query, n=3)
            
            if concept_results:
                for i, (concept, score) in enumerate(concept_results):
                    print(f"  Concept {i+1} (Score: {score:.3f}):")
                    print(f"    Theme: {concept.theme}")
                    print(f"    Summary: {concept.summary[:80]}...")
                    print(f"    Memories: {len(concept.memory_ids)}")
                    print(f"    Key entities: {', '.join(concept.key_entities[:3])}")
                    
                    # Show option to expand concept
                    if i == 0:  # Expand the top result
                        print(f"    \n    📖 Expanding top concept:")
                        memories = self.hippocampus.expand_concept(concept.cluster_id)
                        for j, memory in enumerate(memories[:3]):  # Show first 3 memories
                            print(f"      Memory {j+1}: {memory.text[:60]}...")
            else:
                print("  No matching concepts found.")
    
    def demonstrate_memory_pruning(self):
        """Demonstrate memory analysis and pruning"""
        print(f"\n🌿 Demonstrating Memory Analysis & Pruning...")
        
        # Identify foundational memories
        foundational = self.prefrontal_cortex.identify_foundational_memories(
            min_references=2, 
            min_connections=1
        )
        
        print(f"Foundational Memories ({len(foundational)}):")
        for i, (memory, stats) in enumerate(foundational[:3]):
            print(f"  {i+1}. {memory.text[:60]}...")
            print(f"     Access count: {stats['access_count']}, Connections: {stats['connections']}")
            print(f"     Foundational score: {stats['foundational_score']:.2f}")
        
        # Identify weak memories
        weak = self.prefrontal_cortex.identify_weak_memories(
            max_references=1, 
            max_connections=1,
            min_age_days=0.1  # Very recent for demo
        )
        
        print(f"\nWeak Memories ({len(weak)}):")
        for i, (memory, stats) in enumerate(weak[:3]):
            print(f"  {i+1}. {memory.text[:60]}...")
            print(f"     Access count: {stats['access_count']}, Connections: {stats['connections']}")
            print(f"     Weakness score: {stats['weakness_score']:.2f}")
        
        # Demonstrate memory consolidation (merging similar weak memories)
        print(f"\n🔄 Attempting memory consolidation...")
        merged_count = self.prefrontal_cortex.consolidate_similar_memories(
            similarity_threshold=0.85,
            max_merges=3
        )
        print(f"Merged {merged_count} similar memories")
        
        # Show memory strength distribution
        strength_dist = self.prefrontal_cortex.get_memory_strength_distribution()
        print(f"\nMemory Strength Distribution:")
        print(f"  Foundational: {strength_dist['foundational']}")
        print(f"  Active: {strength_dist['active']}")
        print(f"  Dormant: {strength_dist['dormant']}")
        
        # Show network density
        network_density = self.prefrontal_cortex.calculate_network_density()
        print(f"  Network density: {network_density:.3f}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary of the memory system state"""
        print(f"\n📊 Memory System Summary Report")
        print("=" * 50)
        
        # PrefrontalCortex stats
        pfc_stats = self.prefrontal_cortex.get_memory_stats()
        print(f"Raw Memory Statistics:")
        print(f"  Total memories: {pfc_stats['total_memories']}")
        print(f"  Text memories: {pfc_stats['text_memories']}")
        print(f"  Audio memories: {pfc_stats['audio_memories']}")
        print(f"  Image memories: {pfc_stats['image_memories']}")
        print(f"  Total links: {pfc_stats['total_associative_links']}")
        print(f"  Average links/memory: {pfc_stats['average_links_per_memory']:.2f}")
        
        # Hippocampus stats
        hip_stats = self.hippocampus.get_consolidation_stats()
        print(f"\nConceptual Memory Statistics:")
        print(f"  Total concepts: {hip_stats['total_concepts']}")
        print(f"  Memories in concepts: {hip_stats['total_memories_in_concepts']}")
        print(f"  Consolidation ratio: {hip_stats['consolidation_ratio']:.2%}")
        print(f"  Average concept size: {hip_stats['average_concept_size']:.1f}")
        
        # Memory strength analysis
        strength_dist = self.prefrontal_cortex.get_memory_strength_distribution()
        print(f"\nMemory Strength Distribution:")
        for category, count in strength_dist.items():
            percentage = (count / pfc_stats['total_memories']) * 100 if pfc_stats['total_memories'] > 0 else 0
            print(f"  {category.title()}: {count} ({percentage:.1f}%)")
        
        # Network analysis
        network_density = self.prefrontal_cortex.calculate_network_density()
        print(f"\nNetwork Analysis:")
        print(f"  Network density: {network_density:.3f}")
        print(f"  Temporal chain length: {pfc_stats['temporal_chain_length']}")
        
        # Top concepts by importance
        if self.hippocampus.memory_concepts:
            print(f"\nTop Memory Concepts by Importance:")
            concepts_by_importance = sorted(
                self.hippocampus.memory_concepts.values(),
                key=lambda c: c.importance_score,
                reverse=True
            )
            for i, concept in enumerate(concepts_by_importance[:5]):
                print(f"  {i+1}. {concept.theme} (importance: {concept.importance_score:.2f})")
                print(f"     {len(concept.memory_ids)} memories, {len(concept.key_entities)} entities")


# Update the main demo class
class EnhancedMemorySystemDemo(MemorySystemDemo):
    """Enhanced demo with new functionality"""
    
    # Enhanced demo method
    def demonstrate_enhanced_flow(self):
        """Demonstrate the enhanced memory processing flow"""
        
        print(f"\n🧠 Enhanced Memory Processing Flow Demo")
        print("=" * 50)
        
        test_inputs = [
            "Had a great meeting with Sarah about the new project timeline",
            "Went for lunch with Mike at that Italian place downtown", 
            "Sarah called to discuss the budget concerns for next quarter",
            "Meeting at the office ran late, missed dinner with the family"
        ]
        
        for i, input_text in enumerate(test_inputs):
            print(f"\n--- Processing Input {i+1}: '{input_text}' ---")
            
            # Use enhanced processing flow
            result = self.prefrontal_cortex.process_new_experience(
                text=input_text,
                return_full_context=True
            )
            
            # Display results
            print(f"🔍 Triggered Memories: {len(result['triggered_memories'])}")
            for j, tm in enumerate(result['triggered_memories'][:2]):  # Show top 2
                if 'memory' in tm:
                    print(f"  {j+1}. ({tm['trigger_type']}) {tm['memory'].text[:60]}...")
                    print(f"     Confidence: {tm['confidence']:.3f}")
                elif 'concept' in tm:
                    print(f"  {j+1}. ({tm['trigger_type']}) Concept: {tm['concept'].theme}")
                    print(f"     Confidence: {tm['confidence']:.3f}")
            
            print(f"\n💭 Generated Gist:")
            gist = result['gist']
            if gist['summary']:
                print(f"  Summary: {gist['summary']}")
            if gist['semantic_context']:
                print(f"  Semantic: {gist['semantic_context']}")
            if gist['entity_context']:
                print(f"  Entities: {gist['entity_context']}")
                
            print(f"\n💾 Stored as: {result['new_memory_id'][:8]}...")
            
            # Show full context for last input
            if i == len(test_inputs) - 1 and 'full_context' in result:
                context = result['full_context']
                print(f"\n📊 Full Context Analysis:")
                print(f"  Total triggered: {context['total_triggered']}")
                print(f"  By type: {context['by_type']}")
                print(f"  Key entities: {', '.join(context['key_entities'][:5])}")
        
        # Demonstrate auto-consolidation
        print(f"\n🔄 Running Enhanced Consolidation...")
        consolidation_results = self.hippocampus.auto_consolidate_all(min_age_hours=0.01)
        
        # Show entity statistics
        if hasattr(self.prefrontal_cortex, 'entity_clustering'):
            entity_stats = self.prefrontal_cortex.entity_clustering.get_entity_statistics()
            print(f"\n📈 Entity Statistics:")
            print(f"  Total entities tracked: {entity_stats['total_entities']}")
            print(f"  Most referenced: {', '.join([f'{name}({count})' for name, count in entity_stats['most_referenced_entities'][:5]])}")



    def run_complete_demo(self):
        """Run the complete enhanced demo"""
        
        print("🧠 Enhanced Brain-Like Memory System Demo")
        print("=" * 50)
        
        # 1. Create diverse memories
        memory_ids = self.create_diverse_memories(num_memories=25)
        
        # 2. Demonstrate enhanced processing flow  
        self.demonstrate_enhanced_flow()
        
        # 3. Show all existing demos
        self.demonstrate_memory_recall(num_queries=2)
        self.demonstrate_association_recall()
        self.demonstrate_memory_consolidation()
        self.demonstrate_concept_search()
        self.demonstrate_memory_pruning()
        
        # 4. Generate final summary
        self.generate_summary_report()
        
        print(f"\n✅ Enhanced demo completed successfully!")



if __name__ == "__main__":
    demo = EnhancedMemorySystemDemo()
    demo.run_complete_demo()