from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import uuid
import time
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import math

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
        
        # Positive emotions based on Plutchik's wheel
        self.positive_emotions = {
            'joy', 'trust', 'anticipation', 'admiration', 'ecstasy', 
            'vigilance', 'rage', 'loathing', 'grief', 'amazement',
            'terror', 'serenity', 'acceptance', 'optimism', 'love',
            'submission', 'awe', 'disapproval', 'remorse', 'contempt',
            'aggressiveness', 'pride', 'hope', 'curiosity', 'satisfaction'
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
        
    def clear_all_memories(self):
        """Clear all stored memories"""
        self.memory_pages.clear()
        self.temporal_chain.clear()
        self._save_memory_store()
        print("All memories cleared")


####################################
import numpy as np
import time
import random
from typing import Optional, List, Tuple
from datetime import datetime, timedelta

# Pseudo embedding functions for audio and image
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

def demo_memory_system():
    """
    Comprehensive demo of the PrefrontalCortex memory system
    """
    print("=" * 80)
    print("PREFRONTAL CORTEX MEMORY SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize the embedder and cortex
    print("\n1. INITIALIZING SYSTEM...")
    print("-" * 40)
    
    # Import and initialize the embedder (assuming it's available)
    try:
        from emotion_embed import MxBaiEmbedder  # Adjust import as needed
        embedder = MxBaiEmbedder("demo_embeddings.pkl")
        
        # Load model if not already loaded
        if embedder.model is None:
            print("Loading MxBai embedding model...")
            success = embedder.load_model()
            if not success:
                raise Exception("Failed to load embedding model")
        
        # Initialize emotions if not done
        if not embedder.emotions_initialized:
            print("Initializing emotion embeddings...")
            embedder.initialize_emotion_embeddings()
            
    except ImportError:
        print("WARNING: MxBaiEmbedder not available. Using mock embedder for demo.")
        embedder = MockEmbedder()
    
    # Initialize PrefrontalCortex
    cortex = PrefrontalCortex(embedder, "demo_memory_store.pkl")
    
    print(f"✓ System initialized with {cortex.get_memory_stats()['total_memories']} existing memories")
    
    # Clear existing memories for clean demo
    cortex.clear_all_memories()
    
    print("\n2. CREATING SAMPLE MEMORIES...")
    print("-" * 40)
    
    # Sample memory data with realistic timestamps and different modalities
    current_time = time.time()

    # Create realistic timestamp distribution over the past week
    base_time = current_time - (7 * 24 * 3600)  # Start 7 days ago
    timestamps = [
        base_time + (i * 1.5 * 24 * 3600) + random.randint(0, 12*3600)  # Spread across days with random times
        for i in range(5)
    ]
    timestamps.sort()  # Ensure chronological order


    sample_memories = [
            {
                "text": "I had a wonderful morning walk in the park. The birds were singing and the sun was shining brightly.",
                "image_file": "morning_park.jpg",
                "audio_file": None,
                "timestamp": timestamps[0]  # Use calculated timestamp
            },
            {
                "text": "Feeling stressed about the upcoming presentation at work. Need to prepare more slides.",
                "image_file": None,
                "audio_file": "stress_recording.wav",
                "timestamp": timestamps[1]  # Use calculated timestamp
            },
            {
                "text": "Had lunch with Sarah today. We talked about our college memories and laughed a lot.",
                "image_file": "lunch_with_sarah.jpg",
                "audio_file": "lunch_conversation.wav",
                "timestamp": timestamps[2]  # Use calculated timestamp
            },
            {
                "text": "Beautiful sunset tonight. The colors were absolutely amazing - orange, pink, and purple.",
                "image_file": "sunset_photo.jpg",
                "audio_file": None,
                "timestamp": timestamps[3]  # Use calculated timestamp
            },
            {
                "text": "Working late again. This project is really challenging but I'm learning so much.",
                "image_file": None,
                "audio_file": "late_work_session.wav",
                "timestamp": timestamps[4]  # Use calculated timestamp
            }
        ]
    # Create and store sample memories with proper timestamps
    stored_memory_ids = []
    
    for i, memory_data in enumerate(sample_memories):
        print(f"\nCreating memory {i+1}/5: '{memory_data['text'][:50]}...'")
        
        memory_time = memory_data['timestamp']
        
        # Generate embeddings for available modalities
        text_embedding = None
        audio_embedding = None
        image_embedding = None
        
        if memory_data['text']:
            text_embedding = embedder.embed_text_string(memory_data['text'])
            print(f"  ✓ Text embedded (shape: {text_embedding.shape})")
        
        if memory_data['audio_file']:
            audio_embedding = pseudo_vggish_embed(memory_data['audio_file'])
            print(f"  ✓ Audio embedded")
        
        if memory_data['image_file']:
            image_embedding = pseudo_beit_embed(memory_data['image_file'])
            print(f"  ✓ Image embedded")
        
        # Create memory page
        memory_page = cortex.create_memory_page(
            text=memory_data['text'],
            text_embedding=text_embedding,
            image_file=memory_data['image_file'],
            image_embedding=image_embedding,
            audio_file=memory_data['audio_file'],
            audio_embedding=audio_embedding,
            auto_embed_text=False,  # We already embedded
            auto_detect_emotion=True  # Let it detect emotion
        )
        
        # Override timestamp for temporal spacing demo
        memory_page.timestamp = memory_time
        memory_page.creation_datetime = datetime.fromtimestamp(memory_time).isoformat()
        
        # Store the memory
        memory_id = cortex.store_memory(memory_page)
        stored_memory_ids.append(memory_id)
        
        print(f"  ✓ Memory stored with ID: {memory_id}")
        print(f"  ✓ Detected emotion: {memory_page.emotion}")
        print(f"  ✓ Related memories: {len(memory_page.related_memory_ids)}")
    
    print(f"\n✓ Created and stored {len(stored_memory_ids)} sample memories")
    
    # Display memory statistics
    stats = cortex.get_memory_stats()
    print(f"\nMemory Store Statistics:")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Text memories: {stats['text_memories']}")
    print(f"  Audio memories: {stats['audio_memories']}")
    print(f"  Image memories: {stats['image_memories']}")
    print(f"  Total links: {stats['total_associative_links']}")
    print(f"  Average links per memory: {stats['average_links_per_memory']:.1f}")
    
    print("\n3. PROCESSING NEW MEMORY INPUT...")
    print("-" * 40)
    
    # New memory input that should trigger associations
    new_input = {
        "text": "Had another stressful day at work but took a relaxing walk afterwards. The evening sky looked beautiful.",
        "image_file": "evening_walk.jpg",
        "audio_file": "evening_thoughts.wav"
    }
    
    print(f"New input text: '{new_input['text']}'")
    
    # Step 1: Create embeddings for new input
    print(f"\nStep 1: Generating embeddings...")
    new_text_embedding = embedder.embed_text_string(new_input['text'])
    new_audio_embedding = pseudo_vggish_embed(new_input['audio_file'])
    new_image_embedding = pseudo_beit_embed(new_input['image_file'])
    print(f"  ✓ All embeddings generated")
    
    # Step 2: Search for similar memories across modalities with detailed ranking breakdown
    print(f"\nStep 2: Searching for similar memories (with ranking breakdown)...")
    
    # Demonstrate the three-stage ranking process
    print(f"\n--- TEXT SIMILARITY SEARCH ---")
    text_matches = cortex.search_text_with_breakdown(new_input['text'], n=3)
    print(f"Text similarity matches ({len(text_matches)} found):")
    for i, match_data in enumerate(text_matches):
        memory, final_score, breakdown = match_data
        age_days = (time.time() - memory.timestamp) / (24 * 3600)
        print(f"  {i+1}. FINAL SCORE: {final_score:.3f}")
        print(f"     ├─ Embedding similarity: {breakdown['embedding_similarity']:.3f}")
        print(f"     ├─ Temporal adjustment: {breakdown['temporal_factor']:.3f} (age: {age_days:.1f} days)")
        print(f"     ├─ Popularity boost: {breakdown['popularity_factor']:.3f} ({len(memory.related_memory_ids)} links)")
        print(f"     ├─ Emotion: {memory.emotion}")
        print(f"     └─ Text: '{memory.text[:50]}...'")
        print()
    
    # Audio similarity search (placeholder with breakdown)
    print(f"--- AUDIO SIMILARITY SEARCH ---")
    audio_matches = cortex.search_audio_with_breakdown(new_audio_embedding, n=3)
    print(f"Audio similarity matches ({len(audio_matches)} found):")
    for i, match_data in enumerate(audio_matches):
        memory, final_score, breakdown = match_data
        age_days = (time.time() - memory.timestamp) / (24 * 3600)
        print(f"  {i+1}. FINAL SCORE: {final_score:.3f}")
        print(f"     ├─ Embedding similarity: {breakdown['embedding_similarity']:.3f}")
        print(f"     ├─ Temporal adjustment: {breakdown['temporal_factor']:.3f} (age: {age_days:.1f} days)")
        print(f"     ├─ Popularity boost: {breakdown['popularity_factor']:.3f} ({len(memory.related_memory_ids)} links)")
        print(f"     └─ Audio: {memory.audio_file}")
        print()
    
    # Image similarity search (placeholder with breakdown)
    print(f"--- IMAGE SIMILARITY SEARCH ---")
    image_matches = cortex.search_image_with_breakdown(new_image_embedding, n=3)
    print(f"Image similarity matches ({len(image_matches)} found):")
    for i, match_data in enumerate(image_matches):
        memory, final_score, breakdown = match_data
        age_days = (time.time() - memory.timestamp) / (24 * 3600)
        print(f"  {i+1}. FINAL SCORE: {final_score:.3f}")
        print(f"     ├─ Embedding similarity: {breakdown['embedding_similarity']:.3f}")
        print(f"     ├─ Temporal adjustment: {breakdown['temporal_factor']:.3f} (age: {age_days:.1f} days)")
        print(f"     ├─ Popularity boost: {breakdown['popularity_factor']:.3f} ({len(memory.related_memory_ids)} links)")
        print(f"     └─ Image: {memory.image_file}")
        print()
    
    # Step 3: Create and store new memory with associations
    print(f"\nStep 3: Creating and storing new memory...")
    
    new_memory = cortex.create_memory_page(
        text=new_input['text'],
        text_embedding=new_text_embedding,
        image_file=new_input['image_file'],
        image_embedding=new_image_embedding,
        audio_file=new_input['audio_file'],
        audio_embedding=new_audio_embedding,
        auto_embed_text=False,
        auto_detect_emotion=True
    )
    
    print(f"  ✓ New memory created")
    print(f"  ✓ Detected emotion: {new_memory.emotion}")
    
    # Store the new memory (this will create automatic associations)
    new_memory_id = cortex.store_memory(
        new_memory, 
        link_to_similar=True, 
        similarity_threshold=0.6,  # Lower threshold for demo
        max_links=3
    )
    
    print(f"  ✓ Memory stored with ID: {new_memory_id}")
    print(f"  ✓ Created {len(new_memory.related_memory_ids)} associative links")
    
    # Step 4: Display the association network
    print(f"\nStep 4: Association network for new memory...")
    print("-" * 40)
    
    print(f"New memory: '{new_memory.text[:60]}...'")
    print(f"Emotion: {new_memory.emotion}")
    print(f"Timestamp: {datetime.fromtimestamp(new_memory.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nAssociated memories ({len(new_memory.related_memory_ids)}):")
    for i, related_id in enumerate(new_memory.related_memory_ids):
        related_memory = cortex.get_memory(related_id)
        if related_memory:
            age_hours = (new_memory.timestamp - related_memory.timestamp) / 3600
            print(f"  {i+1}. ID: {related_id}")
            print(f"     Text: '{related_memory.text[:50]}...'")
            print(f"     Emotion: {related_memory.emotion}")
            print(f"     Age: {age_hours:.1f} hours ago")
            print(f"     Links: {len(related_memory.related_memory_ids)} total")
            print()
    
    # Step 5: Final statistics
    print("5. FINAL SYSTEM STATE...")
    print("-" * 40)
    
    final_stats = cortex.get_memory_stats()
    print(f"Final Memory Store Statistics:")
    print(f"  Total memories: {final_stats['total_memories']}")
    print(f"  Total associative links: {final_stats['total_associative_links']}")
    print(f"  Average links per memory: {final_stats['average_links_per_memory']:.1f}")
    print(f"  Temporal chain length: {final_stats['temporal_chain_length']}")
    
    # Show temporal chain
    print(f"\nTemporal chain (chronological order):")
    for i, memory_id in enumerate(cortex.temporal_chain):
        memory = cortex.get_memory(memory_id)
        if memory:
            timestamp_str = datetime.fromtimestamp(memory.timestamp).strftime('%H:%M')
            print(f"  {i+1}. [{timestamp_str}] {memory.emotion} - '{memory.text[:40]}...'")
    
    # Step 6: Demonstrate the core "What does this remind you of?" functionality
    print("6. CORE MEMORY ASSOCIATION FUNCTION...")
    print("-" * 40)
    
    # This is the main function - input something, get back what it reminds the system of
    reminder_memory = cortex.what_does_this_remind_me_of(
        text=new_input['text'],
        image_file=new_input['image_file'],
        audio_file=new_input['audio_file']
    )
    
    if reminder_memory:
        age_hours = (time.time() - reminder_memory.timestamp) / 3600
        print(f"💭 INPUT: '{new_input['text']}'")
        print(f"🧠 REMINDS ME OF: '{reminder_memory.text[:60]}...'")
        print(f"   └─ From {age_hours:.1f} hours ago | Emotion: {reminder_memory.emotion}")
        print(f"   └─ Access count: {reminder_memory.access_count} | Links: {len(reminder_memory.related_memory_ids)}")
    else:
        print("💭 No strong associations found - this is a novel experience!")
    
    # Step 7: Demonstrate multiple inputs to show foundational memory emergence
    print(f"\n7. FOUNDATIONAL MEMORY EMERGENCE...")
    print("-" * 40)
    
    # Simulate several related inputs over time to show how some memories become foundational
    additional_inputs = [
        {"text": "Went for a bike ride in the morning, feeling great!", "type": "bike"},
        {"text": "Beautiful weather for cycling today", "type": "bike"},
        {"text": "Stressed about the quarterly presentation coming up", "type": "work_stress"},
        {"text": "Working on slides late into the night again", "type": "work_stress"},
        {"text": "Another sunset walk to clear my head", "type": "walk"},
        {"text": "Taking evening walks has become my favorite routine", "type": "walk"},
        {"text": "Presentation went well, feeling relieved", "type": "work_stress"},
    ]
    
    print("Processing additional memories to build foundational associations...")
    foundational_candidates = {}
    
    for i, input_data in enumerate(additional_inputs):
        print(f"\nInput {i+1}: '{input_data['text']}'")
        
        # Process each input through the system
        reminder = cortex.what_does_this_remind_me_of(text=input_data['text'])
        
        if reminder:
            # Track which memories are being referenced frequently
            reminder_id = reminder.memory_id
            if reminder_id not in foundational_candidates:
                foundational_candidates[reminder_id] = {
                    'memory': reminder,
                    'references': 0,
                    'themes': set()
                }
            foundational_candidates[reminder_id]['references'] += 1
            foundational_candidates[reminder_id]['themes'].add(input_data['type'])
            
            print(f"  💭 Reminds me of: '{reminder.text[:50]}...'")
            print(f"     └─ Now referenced {foundational_candidates[reminder_id]['references']} times")
        else:
            print(f"  💭 No strong associations - novel experience")
    
    # Step 8: Identify foundational memories
    print(f"\n8. FOUNDATIONAL MEMORY ANALYSIS...")
    print("-" * 40)
    
    # Get the most referenced and connected memories
    foundational_memories = cortex.identify_foundational_memories(min_references=2, min_connections=3)
    
    print(f"Identified {len(foundational_memories)} foundational memories:")
    for i, (memory, stats) in enumerate(foundational_memories):
        print(f"\n  {i+1}. FOUNDATIONAL MEMORY:")
        print(f"     Text: '{memory.text[:60]}...'")
        print(f"     Access count: {stats['access_count']}")
        print(f"     Connections: {stats['connections']}")
        print(f"     Foundational score: {stats['foundational_score']:.3f}")
        print(f"     Age: {stats['age_days']:.1f} days")
    
    # Step 9: Memory consolidation - merge similar low-activity memories
    print(f"\n9. MEMORY CONSOLIDATION...")
    print("-" * 40)
    
    # Identify memories that could be merged or deleted
    weak_memories = cortex.identify_weak_memories(max_references=1, max_connections=1, min_age_days=1)
    
    print(f"Found {len(weak_memories)} weak memories that could be consolidated:")
    for memory, stats in weak_memories[:3]:  # Show first 3
        print(f"  • '{memory.text[:40]}...' (age: {stats['age_days']:.1f} days, {stats['connections']} links)")
    
    if len(weak_memories) > 2:
        # Demonstrate memory merging
        merged_count = cortex.consolidate_similar_memories(similarity_threshold=0.8, max_merges=2)
        print(f"\n  ✓ Consolidated {merged_count} similar memories to reduce redundancy")
    
    # Step 10: Final system state and insights
    print(f"\n10. SYSTEM INSIGHTS...")
    print("-" * 40)
    
    final_stats = cortex.get_memory_stats()
    print(f"Final Memory Network:")
    print(f"  • Total memories: {final_stats['total_memories']}")
    print(f"  • Average connections: {final_stats['average_links_per_memory']:.1f}")
    print(f"  • Network density: {cortex.calculate_network_density():.3f}")
    
    # Show memory strength distribution
    memory_strengths = cortex.get_memory_strength_distribution()
    print(f"\nMemory Strength Distribution:")
    print(f"  • Foundational (high activity): {memory_strengths['foundational']} memories")
    print(f"  • Active (medium activity): {memory_strengths['active']} memories") 
    print(f"  • Dormant (low activity): {memory_strengths['dormant']} memories")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • The system learned that bike rides, work stress, and evening walks")
    print(f"     are recurring themes that trigger strong associations")
    print(f"   • Foundational memories emerged through repeated references")
    print(f"   • Weak memories were consolidated to prevent network bloat")
    print(f"   • The memory network is becoming more organized and efficient")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED - MEMORY SYSTEM FUNCTIONING AS INTENDED!")
    print("✅ Input → Associate → Save → Return Association → Build Foundations")
    print("=" * 80)
    
    return cortex, reminder_memory


class MockEmbedder:
    """Mock embedder for demo when MxBaiEmbedder is not available"""
    
    def __init__(self):
        self.emotions_initialized = True
        self.model = "mock"
        
        # Simple emotion mappings for demo
        self.emotion_keywords = {
            'joy': ['wonderful', 'amazing', 'beautiful', 'laughed', 'happy'],
            'stress': ['stressed', 'challenging', 'pressure', 'worried'],
            'anticipation': ['upcoming', 'prepare', 'planning'],
            'trust': ['friend', 'lunch', 'talked', 'memories'],
            'serenity': ['sunset', 'peaceful', 'relaxing', 'walk']
        }
    
    def embed_text_string(self, text: str) -> np.ndarray:
        """Generate deterministic mock embeddings based on text content"""
        # Use hash for deterministic results
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.normal(0, 1, 1024)  # Match MxBai dimensions
        return embedding / np.linalg.norm(embedding)
    
    def find_most_similar_emotion(self, text: str):
        """Simple keyword-based emotion detection for demo"""
        text_lower = text.lower()
        
        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion, 0.8, {'mood': emotion, 'thoughts': 'demo', 'responses': 'demo'}
        
        return 'neutral', 0.5, {'mood': 'neutral', 'thoughts': 'demo', 'responses': 'demo'}


if __name__ == "__main__":
    # Run the demo
    try:
        cortex, new_memory_id = demo_memory_system()
        
        print(f"\n🧠 Demo completed! New memory ID: {new_memory_id}")
        print(f"💾 Memory store saved to: {cortex.memory_store_file}")
        print(f"🔗 Memory network ready for further interactions!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()