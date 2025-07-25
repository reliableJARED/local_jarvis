"""
Speaker Recognition, Profiling and Embedding System
Handles speaker identification, clustering, and profile management
"""

import numpy as np
import time
from datetime import datetime
from collections import deque
import threading
import pickle
import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine, pdist, squareform
import warnings

warnings.filterwarnings("ignore")

# Try to import speaker embedding models
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

@dataclass
class SpeakerProfile:
    """Enhanced speaker profile with multiple embeddings and metadata"""
    speaker_id: str
    embeddings: List[np.ndarray]  # Store multiple embeddings for better accuracy
    first_seen: datetime
    last_updated: datetime
    total_samples: int
    confidence_scores: List[float]  # Track confidence for each embedding
    
    def get_average_embedding(self) -> np.ndarray:
        """Get average embedding from all samples"""
        if not self.embeddings:
            return np.array([])
        
        # Weight embeddings by confidence if available
        if len(self.confidence_scores) == len(self.embeddings):
            weights = np.array(self.confidence_scores)
            weights = weights / np.sum(weights)  # Normalize
            
            weighted_embeddings = [emb * weight for emb, weight in zip(self.embeddings, weights)]
            return np.mean(weighted_embeddings, axis=0)
        else:
            return np.mean(self.embeddings, axis=0)
    
    def add_embedding(self, embedding: np.ndarray, confidence: float = 1.0):
        """Add a new embedding sample"""
        self.embeddings.append(embedding)
        self.confidence_scores.append(confidence)
        self.total_samples += 1
        self.last_updated = datetime.now()
        
        # Keep only the best N embeddings to prevent memory bloat
        max_embeddings = 10
        if len(self.embeddings) > max_embeddings:
            # Remove embeddings with lowest confidence
            sorted_indices = np.argsort(self.confidence_scores)
            keep_indices = sorted_indices[-max_embeddings:]
            
            self.embeddings = [self.embeddings[i] for i in keep_indices]
            self.confidence_scores = [self.confidence_scores[i] for i in keep_indices]

@dataclass
class SpeakerCluster:
    """Represents a cluster of embeddings for a speaker"""
    cluster_id: int
    embeddings: List[np.ndarray]
    confidence_scores: List[float]
    centroid: np.ndarray
    cluster_quality: float  # Silhouette score or similar
    sample_count: int
    
    def update_centroid(self):
        """Recalculate centroid with confidence weighting"""
        if not self.embeddings:
            return
        
        try:
            # Ensure embeddings are numpy arrays
            embeddings_array = [np.array(emb) if not isinstance(emb, np.ndarray) else emb 
                            for emb in self.embeddings]
            
            if len(self.confidence_scores) == len(embeddings_array):
                weights = np.array(self.confidence_scores)
                if np.sum(weights) > 0:  # Avoid division by zero
                    weights = weights / np.sum(weights)
                    weighted_embeddings = [emb * weight for emb, weight in zip(embeddings_array, weights)]
                    self.centroid = np.mean(weighted_embeddings, axis=0)
                else:
                    self.centroid = np.mean(embeddings_array, axis=0)
            else:
                self.centroid = np.mean(embeddings_array, axis=0)
        except Exception as e:
            print(f"Error updating centroid: {e}")
            # Fallback: just use simple mean if weighting fails
            try:
                embeddings_array = [np.array(emb) if not isinstance(emb, np.ndarray) else emb 
                                for emb in self.embeddings]
                self.centroid = np.mean(embeddings_array, axis=0)
            except:
                self.centroid = np.array([])

@dataclass
class ClusteredSpeakerProfile:
    """Enhanced speaker profile with clustering support"""
    speaker_id: str
    clusters: List[SpeakerCluster]
    first_seen: datetime
    last_updated: datetime
    total_samples: int
    last_clustering: Optional[datetime] = None
    clustering_version: int = 0
    
    def get_primary_embedding(self) -> np.ndarray:
        """Get the best representative embedding"""
        if not self.clusters:
            return np.array([])
        
        # Find cluster with highest quality and most samples
        best_cluster = max(self.clusters, 
                          key=lambda c: c.cluster_quality * np.log(c.sample_count + 1))
        return best_cluster.centroid
    
    def get_all_centroids(self) -> List[np.ndarray]:
        """Get centroids from all clusters"""
        return [cluster.centroid for cluster in self.clusters if len(cluster.centroid) > 0]
    
    def add_embedding(self, embedding: np.ndarray, confidence: float = 1.0):
        """Add embedding to the most appropriate cluster"""
        if not self.clusters:
            # Create first cluster
            cluster = SpeakerCluster(
                cluster_id=0,
                embeddings=[embedding],
                confidence_scores=[confidence],
                centroid=embedding.copy(),
                cluster_quality=1.0,
                sample_count=1
            )
            self.clusters.append(cluster)
        else:
            # Find best matching cluster
            best_cluster = None
            best_similarity = 0.0
            
            for cluster in self.clusters:
                similarity = 1 - cosine(embedding, cluster.centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster
            
            # Add to best cluster if similarity is good enough, otherwise create new cluster
            if best_similarity > 0.7:  # Threshold for same cluster
                best_cluster.embeddings.append(embedding)
                best_cluster.confidence_scores.append(confidence)
                best_cluster.sample_count += 1
                best_cluster.update_centroid()
            else:
                # Create new cluster
                new_cluster_id = max(c.cluster_id for c in self.clusters) + 1
                cluster = SpeakerCluster(
                    cluster_id=new_cluster_id,
                    embeddings=[embedding],
                    confidence_scores=[confidence],
                    centroid=embedding.copy(),
                    cluster_quality=1.0,
                    sample_count=1
                )
                self.clusters.append(cluster)
        
        self.total_samples += 1
        self.last_updated = datetime.now()

class SpeakerEmbeddingExtractor:
    """Handles extraction of speaker embeddings from audio"""
    
    def __init__(self):
        # Load speaker embedding model
        if SPEECHBRAIN_AVAILABLE:
            try:
                self.speaker_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
                self.model_available = True
                print("SpeechBrain speaker model loaded successfully")
            except Exception as e:
                print(f"Failed to load SpeechBrain model: {e}")
                self.speaker_model = None
                self.model_available = False
        else:
            print("SpeechBrain not available, using fallback feature extraction")
            self.speaker_model = None
            self.model_available = False
    
    def extract_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio"""
        if not self.model_available or self.speaker_model is None:
            return self._fallback_features(audio_data)
        
        try:
            # Use SpeechBrain model
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return self._fallback_features(audio_data)
    
    def _fallback_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Fallback: simple audio features when SpeechBrain is not available"""
        if len(audio_data) == 0:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        energy = np.mean(audio_data ** 2)
        fft = np.fft.fft(audio_data)
        spectral_centroid = np.mean(np.abs(fft))
        zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio_data))))
        
        embedding = np.array([energy * 1000, spectral_centroid / 1000, zero_crossing_rate * 100, 0, 0])
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0, 0.0])

class SpeakerIdentifier:
    """Speaker identifier with automatic clustering capabilities"""
    
    def __init__(self, speaker_db_path: str = "clustered_speaker_profiles.pkl",
                 auto_save_interval: int = 300,  # 5 minutes
                 clustering_interval: int = 300,  # 5 minutes
                 auto_clustering: bool = False,  # If true, cluster every clustering_interval seconds
                 min_samples_for_clustering: int = 10):
        
        self.speaker_db_path = speaker_db_path
        self.auto_save_interval = auto_save_interval
        self.clustering_interval = clustering_interval
        self.min_samples_for_clustering = min_samples_for_clustering
        self.auto_clustering = auto_clustering
        self.speaker_profiles: Dict[str, ClusteredSpeakerProfile] = {}
        self.current_speaker_id = None
        self.speaker_counter = 0
        self._save_lock = threading.Lock()
        self._cluster_lock = threading.Lock()
        self._last_save_time = 0
        self._last_clustering_time = 0
        self._speakerid_prefix = "USER"
        
        # Initialize embedding extractor
        self.embedding_extractor = SpeakerEmbeddingExtractor()
        
        # Load existing profiles
        self.load_speaker_profiles()
        
        # Start background threads
        self._auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self._auto_save_thread.start()
        
        self._clustering_thread = threading.Thread(target=self._clustering_loop, daemon=True)
        self._clustering_thread.start()
    
    def extract_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio (delegates to embedding extractor)"""
        return self.embedding_extractor.extract_embedding(audio_data)
    
    def identify_speaker(self, audio_data: np.ndarray, threshold: float = 0.3,
                        update_embedding: bool = True) -> str:
        """Identify speaker using clustered embeddings"""
        embedding = self.extract_embedding(audio_data)
        
        # Find best matching speaker across all their clusters
        best_match_id = None
        best_similarity = 0.0
        
        for speaker_id, profile in self.speaker_profiles.items():
            # Check against all cluster centroids
            for centroid in profile.get_all_centroids():
                if len(centroid) > 0:
                    similarity = 1 - cosine(embedding, centroid)
                    if similarity > best_similarity and similarity > threshold:
                        best_similarity = similarity
                        best_match_id = speaker_id
        
        if best_match_id is not None:
            if update_embedding:
                self.update_speaker_embedding(best_match_id, embedding, best_similarity)
            return best_match_id
        else:
            # Create new speaker
            new_speaker_id = f"{self._speakerid_prefix}_{self.speaker_counter:02d}"
            self.speaker_counter += 1
            self.create_new_speaker(new_speaker_id, embedding)
            return new_speaker_id
    
    def create_new_speaker(self, speaker_id: str, embedding: np.ndarray, confidence: float = 1.0):
        """Create a new speaker profile with initial cluster"""
        profile = ClusteredSpeakerProfile(
            speaker_id=speaker_id,
            clusters=[],
            first_seen=datetime.now(),
            last_updated=datetime.now(),
            total_samples=0
        )
        profile.add_embedding(embedding, confidence)
        self.speaker_profiles[speaker_id] = profile
        print(f"Created new clustered speaker profile: {speaker_id}")
    
    def rename_speaker(self, speaker_id: str, new_name: str):
        """
        Rename a speaker with collision detection and formatting
        
        Args:
            speaker_id: Current speaker ID to rename
            new_name: New name for the speaker (e.g., 'Bob', 'Bob Johnson')
        
        Returns:
            str: The final speaker_id that was assigned, or None if operation failed
        """
        # Check if the speaker exists
        if speaker_id not in self.speaker_profiles:
            print(f"Error: Speaker '{speaker_id}' not found")
            return None
        
        # Format the new name as a valid speaker_id
        # Replace spaces with underscores and clean up the name
        formatted_name = new_name.strip().replace(' ', '_')
        
        # Remove any characters that might cause issues (keep alphanumeric, underscore, hyphen)
        import re
        formatted_name = re.sub(r'[^a-zA-Z0-9_\-]', '', formatted_name)
        
        if not formatted_name:
            print("Error: Invalid name provided")
            return None
        
        # Handle collision detection
        final_speaker_id = self._get_unique_speaker_id(formatted_name)
        
        # If the final ID is the same as current, no change needed
        if final_speaker_id == speaker_id:
            print(f"Speaker '{speaker_id}' name unchanged")
            return speaker_id
        
        # Perform the rename by moving the profile to the new key
        try:
            # Get the existing profile
            profile = self.speaker_profiles[speaker_id]
            
            # Update the speaker_id within the profile
            profile.speaker_id = final_speaker_id
            profile.last_updated = datetime.now()
            
            # Move to new key in the dictionary
            self.speaker_profiles[final_speaker_id] = profile
            
            # Remove the old key
            del self.speaker_profiles[speaker_id]
            
            # Update current_speaker_id if it was pointing to the renamed speaker
            if self.current_speaker_id == speaker_id:
                self.current_speaker_id = final_speaker_id
            
            print(f"Successfully renamed speaker '{speaker_id}' to '{final_speaker_id}'")
            
            # Schedule a save to persist the change
            self._schedule_save()
            
            return final_speaker_id
            
        except Exception as e:
            print(f"Error renaming speaker: {e}")
            return None

    def _get_unique_speaker_id(self, base_name: str) -> str:
        """
        Generate a unique speaker ID by handling name collisions
        
        Args:
            base_name: The desired base name (e.g., 'Bob', 'Bob_Johnson')
        
        Returns:
            str: A unique speaker ID (e.g., 'Bob', 'Bob_2', 'Bob_Johnson_3')
        """
        # Check if the base name is already unique
        if base_name not in self.speaker_profiles:
            return base_name
        
        # Find a unique variation by appending numbers
        counter = 2
        while True:
            candidate_name = f"{base_name}_{counter}"
            if candidate_name not in self.speaker_profiles:
                return candidate_name
            counter += 1
            
            # Safety check to prevent infinite loops
            if counter > 1000:
                # Fallback to timestamp-based uniqueness
                timestamp_suffix = str(int(time.time()))[-6:]  # Last 6 digits of timestamp
                return f"{base_name}_{timestamp_suffix}"

    def update_speaker_embedding(self, speaker_id: str, embedding: np.ndarray, confidence: float = 1.0):
        """Update speaker embedding and trigger clustering if needed"""
        if speaker_id in self.speaker_profiles:
            profile = self.speaker_profiles[speaker_id]
            profile.add_embedding(embedding, confidence)
            
            # Check if this speaker needs re-clustering
            if (profile.total_samples >= self.min_samples_for_clustering and
                profile.total_samples % 20 == 0):  # Re-cluster every 20 new samples
                self._schedule_speaker_clustering(speaker_id)
    
    def _schedule_speaker_clustering(self, speaker_id: str):
        """Schedule clustering for a specific speaker"""
        def cluster_task():
            self._cluster_speaker(speaker_id)
        
        threading.Thread(target=cluster_task, daemon=True).start()
    
    def _cluster_speaker(self, speaker_id: str):
        """Perform clustering analysis for a specific speaker"""
        if speaker_id not in self.speaker_profiles:
            return
        
        with self._cluster_lock:
            profile = self.speaker_profiles[speaker_id]
            
            # Collect all embeddings from all clusters
            all_embeddings = []
            all_confidences = []
            
            for cluster in profile.clusters:
                all_embeddings.extend(cluster.embeddings)
                all_confidences.extend(cluster.confidence_scores)
            
            if len(all_embeddings) < self.min_samples_for_clustering:
                return
            
            print(f"Clustering {len(all_embeddings)} embeddings for {speaker_id}")
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings)
            confidences_array = np.array(all_confidences)
            
            # Try different clustering approaches
            best_clusters = self._find_optimal_clusters(embeddings_array, confidences_array)
            
            if best_clusters:
                # Update speaker profile with new clusters
                new_clusters = []
                for i, (cluster_embeddings, cluster_confidences) in enumerate(best_clusters):
                    cluster = SpeakerCluster(
                        cluster_id=i,
                        embeddings=cluster_embeddings,
                        confidence_scores=cluster_confidences,
                        centroid=np.array([]),  # Will be calculated
                        cluster_quality=0.0,
                        sample_count=len(cluster_embeddings)
                    )
                    cluster.update_centroid()
                    
                    # Calculate cluster quality (compactness)
                    if len(cluster_embeddings) > 1:
                        distances = pdist([cluster.centroid] + cluster_embeddings, metric='cosine')
                        cluster.cluster_quality = 1.0 - np.mean(distances)
                    else:
                        cluster.cluster_quality = 1.0
                    
                    new_clusters.append(cluster)
                
                profile.clusters = new_clusters
                profile.last_clustering = datetime.now()
                profile.clustering_version += 1
                
                print(f"Updated {speaker_id} with {len(new_clusters)} clusters")
    
    def _find_optimal_clusters(self, embeddings: np.ndarray, confidences: np.ndarray) -> List[Tuple[List, List]]:
        """Find optimal clustering for embeddings"""
        if len(embeddings) < 3:
            return [(embeddings.tolist(), confidences.tolist())]
        
        best_score = -1
        best_clusters = None
        
        # Try DBSCAN clustering
        try:
            # Use DBSCAN with cosine distance
            distance_matrix = squareform(pdist(embeddings, metric='cosine'))
            dbscan = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
            cluster_labels = dbscan.fit_predict(distance_matrix)
            
            # Check if clustering is good
            if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
                score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_clusters = self._group_by_labels(embeddings, confidences, cluster_labels)
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")
        
        # Try K-means with different k values
        for k in range(2, min(5, len(embeddings) // 2 + 1)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Calculate silhouette score
                score = silhouette_score(embeddings, cluster_labels, metric='cosine')
                if score > best_score:
                    best_score = score
                    best_clusters = self._group_by_labels(embeddings, confidences, cluster_labels)
            except Exception as e:
                print(f"K-means clustering (k={k}) failed: {e}")
        
        # If no good clustering found, return single cluster
        if best_clusters is None or best_score < 0.3:
            return [(embeddings.tolist(), confidences.tolist())]
        
        return best_clusters
    
    def _group_by_labels(self, embeddings: np.ndarray, confidences: np.ndarray, 
                        labels: np.ndarray) -> List[Tuple[List, List]]:
        """Group embeddings by cluster labels"""
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue
            
            mask = labels == label
            cluster_embeddings = embeddings[mask].tolist()
            cluster_confidences = confidences[mask].tolist()
            clusters.append((cluster_embeddings, cluster_confidences))
        
        return clusters
    
    def _clustering_loop(self):
        """Background thread for periodic clustering"""
        while True:
            time.sleep(self.clustering_interval)
            current_time = time.time()
            
            if current_time - self._last_clustering_time > self.clustering_interval:
                if self.auto_clustering:
                    print(f"AUTO CLUSTERING: {self.auto_clustering}")
                    self._perform_global_clustering()
                    self._last_clustering_time = current_time
    
    def _perform_global_clustering(self):
        """Perform clustering analysis on all speakers that need it"""
        speakers_to_cluster = []
        print("Clustering saved speakers")
        for speaker_id, profile in self.speaker_profiles.items():
            # Check if speaker needs clustering
            needs_clustering = (
                profile.total_samples >= self.min_samples_for_clustering and
                (profile.last_clustering is None or 
                 (datetime.now() - profile.last_clustering).seconds > self.clustering_interval * 2)
            )
            
            if needs_clustering:
                speakers_to_cluster.append(speaker_id)
        
        if speakers_to_cluster:
            print(f"Performing clustering analysis on {len(speakers_to_cluster)} speakers")
            for speaker_id in speakers_to_cluster:
                self._cluster_speaker(speaker_id)
    
    def get_clustering_stats(self) -> Dict[str, Dict]:
        """Get clustering statistics for all speakers"""
        stats = {}
        for speaker_id, profile in self.speaker_profiles.items():
            cluster_info = []
            for cluster in profile.clusters:
                cluster_info.append({
                    'cluster_id': cluster.cluster_id,
                    'sample_count': cluster.sample_count,
                    'quality': cluster.cluster_quality,
                    'avg_confidence': np.mean(cluster.confidence_scores) if cluster.confidence_scores else 0.0
                })
            
            stats[speaker_id] = {
                'total_samples': profile.total_samples,
                'num_clusters': len(profile.clusters),
                'last_clustering': profile.last_clustering,
                'clustering_version': profile.clustering_version,
                'clusters': cluster_info
            }
        
        return stats
    
    def merge_speakers(self, speaker_id1: str, speaker_id2: str, keep_id: str = None) -> str:
        """
        Merge two speaker profiles (useful for correcting misidentifications)
        
        Args:
            speaker_id1, speaker_id2: Speaker IDs to merge
            keep_id: Which ID to keep (if None, keeps speaker_id1)
        
        Returns:
            The retained speaker ID
        """
        if speaker_id1 not in self.speaker_profiles or speaker_id2 not in self.speaker_profiles:
            raise ValueError("One or both speakers not found")
        
        profile1 = self.speaker_profiles[speaker_id1]
        profile2 = self.speaker_profiles[speaker_id2]
        
        # Determine which profile to keep
        if keep_id is None:
            keep_id = speaker_id1
        elif keep_id not in [speaker_id1, speaker_id2]:
            raise ValueError("keep_id must be one of the speakers being merged")
        
        # Merge clusters instead of just embeddings
        if keep_id == speaker_id1:
            profile1.clusters.extend(profile2.clusters)
            profile1.total_samples += profile2.total_samples
            profile1.last_updated = max(profile1.last_updated, profile2.last_updated)
            # Remove the other profile
            del self.speaker_profiles[speaker_id2]
            return speaker_id1
        else:
            profile2.clusters.extend(profile1.clusters)
            profile2.total_samples += profile1.total_samples
            profile2.last_updated = max(profile1.last_updated, profile2.last_updated)
            # Remove the other profile
            del self.speaker_profiles[speaker_id1]
            return speaker_id2
    
    def get_speaker_stats(self) -> Dict[str, Dict]:
        """Get statistics about all known speakers"""
        stats = {}
        for speaker_id, profile in self.speaker_profiles.items():
            # Calculate average confidence across all clusters
            all_confidences = []
            for cluster in profile.clusters:
                all_confidences.extend(cluster.confidence_scores)
            
            stats[speaker_id] = {
                'total_samples': profile.total_samples,
                'num_clusters': len(profile.clusters),
                'first_seen': profile.first_seen,
                'last_updated': profile.last_updated,
                'avg_confidence': np.mean(all_confidences) if all_confidences else 0.0
            }
        return stats
    
    def save_speaker_profiles(self, filepath: str = None):
        """Save speaker profiles to pickle file"""
        if filepath is None:
            filepath = self.speaker_db_path
        
        with self._save_lock:
            try:
                # Create backup of existing file
                if os.path.exists(filepath):
                    backup_path = f"{filepath}.backup"
                    os.rename(filepath, backup_path)
                
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        'speaker_profiles': self.speaker_profiles,
                        'speaker_counter': self.speaker_counter,
                        'save_timestamp': datetime.now()
                    }, f)
                
                # Remove backup if save was successful
                backup_path = f"{filepath}.backup"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                
                self._last_save_time = time.time()
                print(f"Saved {len(self.speaker_profiles)} speaker profiles to {filepath}")
                
            except Exception as e:
                print(f"Error saving speaker profiles: {e}")
                # Restore backup if save failed
                backup_path = f"{filepath}.backup"
                if os.path.exists(backup_path):
                    os.rename(backup_path, filepath)
    
    def load_speaker_profiles(self, filepath: str = None):
        """Load speaker profiles from pickle file"""
        if filepath is None:
            filepath = self.speaker_db_path
        
        if not os.path.exists(filepath):
            print(f"No existing speaker database found at {filepath}")
            return
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.speaker_profiles = data.get('speaker_profiles', {})
            self.speaker_counter = data.get('speaker_counter', 0)
            save_timestamp = data.get('save_timestamp', 'unknown')
            
            print(f"Loaded {len(self.speaker_profiles)} speaker profiles from {filepath}")
            print(f"Database last saved: {save_timestamp}")
            
            # Print speaker stats
            for speaker_id, profile in self.speaker_profiles.items():
                print(f"  {speaker_id}: {profile.total_samples} samples, "
                      f"last updated {profile.last_updated}")
            
        except Exception as e:
            print(f"Error loading speaker profiles: {e}")
            self.speaker_profiles = {}
            self.speaker_counter = 0
    
    def _schedule_save(self):
        """Schedule an asynchronous save"""
        def save_task():
            self.save_speaker_profiles()
        
        threading.Thread(target=save_task, daemon=True).start()
    
    def _auto_save_loop(self):
        """Background thread for periodic auto-saves"""
        while True:
            time.sleep(self.auto_save_interval)
            if time.time() - self._last_save_time > self.auto_save_interval:
                if self.speaker_profiles:  # Only save if we have data
                    self._schedule_save()
    
    def cleanup(self):
        """Clean up and save before shutdown"""
        self.save_speaker_profiles()